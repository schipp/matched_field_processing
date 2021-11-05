import argparse
import logging
import os
import time
from shutil import copyfile

import instaseis
import numpy as np
import obspy
import yaml
from global_land_mask.globe import is_land
from scipy.fftpack import fftfreq
from tqdm import tqdm

from lib.beam import get_beampowers
from lib.data import get_data_spectra
from lib.geometry import (
    generate_global_grid,
    get_all_distances_rounded,
    get_station_locations,
)
from lib.gf import get_gf_spectra_for_dists
from lib.math import get_csdm, svd_csdm
from lib.misc import check_settings_is_valid, settings_gen
from lib.plotting import plot_results
from lib.synth import get_synth_spectra
from lib.time import get_start_times

if __name__ == "__main__":
    # -- CONFIG
    run_time = int(time.time())

    # load settings
    with open("settings.yml", "r") as stream:
        settings = yaml.safe_load(stream)

    check_settings_is_valid(settings=settings)

    # parse whether output should be verbose
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-v", "--verbose", action="store_true")
    # args = parser.parse_args()
    # settings["verbose"] = False
    # if args.verbose:
    #     settings["verbose"] = True

    # project dirs and create if necessary
    project_dir = f"{settings['project_basedir']}/{settings['project_id']}/"
    if not os.path.exists(f"{project_dir}/"):
        os.makedirs(f"{project_dir}/plots/")
        os.makedirs(f"{project_dir}/out/")
    settings["project_dir"] = project_dir

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(message)s",
        filename=f"{project_dir}/{run_time}.log",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # save settings.yml into project dir for future reference
    copyfile(f"settings.yml", f"{project_dir}/{run_time}_settings.yml")
    logging.info(f"Copied settings.yml into {project_dir}/{run_time}_settings.yml")

    # -- TIME
    # define time windows
    start_times = np.array(get_start_times(settings))
    # HACK: append an empty list to only do first time window
    if settings["do_only_one_timewindow"]:
        start_times = start_times[:1].tolist()
        start_times.append([])
    logging.info(f"{start_times=}")

    # -- DATA
    if settings["do_synth"] and settings["use_synth_stations"]:
        st = None
    else:
        logging.info("Reading data...")
        st = obspy.Stream()
        for data_fn in settings["data_fn"]:
            st += obspy.read(data_fn)
        st.merge(fill_value=0)
        logging.info("Data after reading:")
        logging.info(st)

        logging.info(f"Checking duplicate stations...")
        # keep only one of duplicate entries (first one)
        station_names = [tr.stats.station for tr in st]
        unique_stations, unique_counts = np.unique(station_names, return_counts=True)
        duplicate_stations = unique_stations[unique_counts > 1]
        already_saved_once = {tr.stats.station: False for tr in st}
        st_tmp = obspy.Stream()
        for tr in st:
            if (
                tr.stats.station in duplicate_stations
                and already_saved_once[tr.stats.station]
            ):
                continue
            if tr.stats.station in duplicate_stations:
                already_saved_once[tr.stats.station] = True
            st_tmp += tr
        st = st_tmp
        logging.info(f"Duplicate stations: {duplicate_stations}")
        logging.info(f"Data after duplicate removal: {st}")

        # check if stations are outside of any timewindows and remove (for all) if so
        # TODO: make smarter gf_spectra selection to have timewindow-specific stations
        logging.info(f"Checking for missing data in timewindows...")
        removed_traces_all_windows = np.array([])
        n_stations_before_trim = len(st)
        for start_time in tqdm(
            start_times, desc="checking time windows for missing data"
        ):
            if isinstance(start_time, list):
                continue
            st_curr = st.slice(
                starttime=start_time, endtime=start_time + settings["window_length"]
            )
            n_stations_after_trim = len(st_curr)
            if n_stations_after_trim < n_stations_before_trim:
                stations_in_st_curr = [tr.stats.station for tr in st_curr]
                removed_traces = [
                    tr.stats.station
                    for tr in st
                    if tr.stats.station not in stations_in_st_curr
                ]
                removed_traces_all_windows = np.append(
                    removed_traces_all_windows, removed_traces
                )
        unique_removed_station_names = np.unique(removed_traces_all_windows)
        logging.warning(
            f"Removing following stations ({len(unique_removed_station_names)}/{len(st)}): {unique_removed_station_names}"
        )
        st_tmp = obspy.Stream()
        for tr in st:
            if tr.stats.station in unique_removed_station_names:
                continue
            st_tmp += tr
        st = st_tmp

    # -- REPLOT?
    # If results already exist, only replot results.
    one_missing = False
    settings_generator = settings_gen(start_times, settings)
    for start_time, fp, wavetype, n_svd, noise_idx, sdr in settings_generator:
        if isinstance(start_time, list):
            continue
        filename = f"{project_dir}/out/out_{start_time.timestamp}_{settings['window_length']}_{fp}_{n_svd}_{noise_idx}_{sdr}.npy"
        if os.path.isfile(f"{filename}"):
            logging.info(f"Output {filename} already exists. Re-plotting.")
            if settings["do_plot"]:
                logging.info(f"Re-plotting.")
                grid_points, grid_lon_coords, grid_lat_coords = generate_global_grid(
                    settings=settings
                )
                beampowers = np.load(f"{filename}", allow_pickle=True)
                station_locations = get_station_locations(settings, st)
                plot_results(
                    beampowers=beampowers,
                    settings=settings,
                    start_time=start_time,
                    grid_lon_coords=grid_lon_coords,
                    grid_lat_coords=grid_lat_coords,
                    station_locations=station_locations,
                    plot_identifier=f"{fp[0]}_{fp[1]}_{wavetype}_{n_svd}_{noise_idx}",
                )
        else:
            one_missing = True
    if not one_missing:
        import sys

        sys.exit("All maps re-plotted. Exiting")
    logging.info("At least one start_time missing - proceeding")

    # -- GEOMETRY
    grid_points, grid_lon_coords, grid_lat_coords = generate_global_grid(
        settings=settings
    )
    logging.info(f"{grid_points.shape=}")
    station_locations = get_station_locations(settings, st)
    logging.info(f"{station_locations.shape=}")

    # check which grid points are on land to ignore later, if we dont care about them
    if settings["do_synth"]:
        gp_on_land = np.zeros(len(grid_points), dtype=bool)
        if settings["geometry_type"] == "geographic":
            gp_on_land = is_land(grid_points[:, 1], grid_points[:, 0])
    else:
        gp_on_land = is_land(grid_points[:, 1], grid_points[:, 0])

    # compute all distance-azimuth-combinations of gridpoints-station geometry
    gp_dists = get_all_distances_rounded(
        station_locations=station_locations, grid_points=grid_points, settings=settings,
    )
    logging.info(f"{gp_dists.shape=}")

    # azimuths irrelevant if only Z component, then set all to 0
    # can reduce cost massively
    # only valied if MT used is explosion
    if settings["components"] == ["Z"] and settings["MT"] == [1, 1, 1, 0, 0, 0]:
        gp_dists[:, :, 1] = 0

    # extract relevant distances and azimuths only
    relevant_dists_azs = np.unique(
        gp_dists.reshape(len(station_locations) * len(grid_points), 2), axis=0
    )
    logging.info(f"{relevant_dists_azs.shape=}")

    # -- GREEN'S FUNCTIONS
    # intialize green's function database
    instaseis_db = instaseis.open_db(settings["gf_db_dir"])
    logging.info(f"Initialized Instaseis DB at {settings['gf_db_dir']}")

    # compute frequencies for data
    n_samples = int(settings["window_length"] * settings["sampling_rate"])
    freqs = fftfreq(n_samples, 1 / settings["sampling_rate"])

    # compute gf spectra for rounded and non-duplicate gridpoint-station-distances and -azimuths
    # only needed once, if source mechanism never changes
    if not settings["do_svd"]:
        gf_spectra_all = get_gf_spectra_for_dists(
            freqs=freqs,
            dists_azs=relevant_dists_azs,
            instaseis_db=instaseis_db,
            settings=settings,
        )
        logging.info(f"{gf_spectra_all.shape=}")

    # -- Main program loop
    beampowers_per_start_time = []
    settings_generator = settings_gen(start_times, settings)
    for start_time, fp, wavetype, n_svd, noise_idx, sdr in tqdm(
        settings_generator,
        desc="current job",
        total=len(list(settings_gen(start_times, settings))),
    ):
        logging.info(
            f"Starting: {start_time} - {fp}Hz - {wavetype} - {n_svd} eigenvectors - {noise_idx} noise iteration - {sdr}"
        )

        # HACK to skip empty entry when doing only first start time (see above)
        if isinstance(start_time, list):
            continue

        # if grid-searching sdr, need to recompute GF each time!
        if sdr:
            gf_spectra_all = get_gf_spectra_for_dists(
                freqs=freqs,
                dists_azs=relevant_dists_azs,
                instaseis_db=instaseis_db,
                settings=settings,
                sdr=sdr,
            )

        # increase computational speed by excluding frequencies outside of specified band
        if isinstance(fp, list):
            freqs_of_interest_idx = (freqs >= fp[0]) & (freqs <= fp[1])
            # limit gf spectra to frequencies of interest
            gf_spectra = gf_spectra_all[:, freqs_of_interest_idx]
        else:
            freqs_of_interest_idx = None
            gf_spectra = gf_spectra_all
        logging.info(f"After limiting to relevant freqs: {gf_spectra.shape=}")

        filename = f"{project_dir}/out/out_{start_time.timestamp}_{settings['window_length']}_{fp}_{n_svd}_{noise_idx}_{sdr}.npy"
        if os.path.isfile(filename):
            logging.warning(f"output file {filename} exists - skipping")
            continue

        # get spectra for current time window to compare against
        if settings["do_synth"]:
            logging.info("Starting synth spectra computation")
            data_spectra = get_synth_spectra(
                station_locations=station_locations,
                settings=settings,
                instaseis_db=instaseis_db,
                freqs_of_interest_idx=freqs_of_interest_idx,
            )
            data_traces = []
        else:
            data_spectra = get_data_spectra(
                st=st,
                start_time=start_time,
                settings=settings,
                freqs_of_interest_idx=freqs_of_interest_idx,
                fp=fp,
            )
        logging.info(f"{data_spectra.shape=}")

        # ensure spectra shape of synthetic and recorded data is the same
        if data_spectra.shape[1] > gf_spectra.shape[1]:
            error_msg = f"data_spectra too long: {data_spectra.shape[1]=} vs. {gf_spectra.shape[1]=}. SHOULD NEVER HAPPEN"
            logging.critical(error_msg)
            raise ValueError(error_msg)
        elif data_spectra.shape[1] < gf_spectra.shape[1]:
            # For now handle by deleting highest frequency.
            logging.warning(
                f"data_spectra too short: {data_spectra.shape[1]=} vs. {gf_spectra.shape[1]=}. Removing highest {gf_spectra.shape[1] - data_spectra.shape[1]} frequencies."
            )
            gf_spectra = gf_spectra[:, : data_spectra.shape[1]]
        logging.info(f"{gf_spectra.shape=}")

        # compute cross-spectral density matrix
        csdm = get_csdm(spectra=data_spectra)
        logging.info(f"{csdm.shape=}")

        if n_svd > 0 and settings["do_svd"]:
            logging.info(
                f"SVD reduction starting from {n_svd}/{csdm.shape[0]} components"
            )
            csdm = svd_csdm(csdm, n_svd, settings)
            logging.info(f"{csdm.shape=}")

        beampowers = get_beampowers(
            csdm=csdm,
            gf_spectra=gf_spectra,
            gp_dists=gp_dists,
            gp_on_land=gp_on_land,
            relevant_dists=relevant_dists_azs,
            settings=settings,
        )

        # reshape output into matrix and save
        beampowers = np.array(beampowers).reshape(
            len(grid_lon_coords), len(grid_lat_coords)
        )
        np.save(filename, beampowers)

        if settings["do_plot"]:
            plot_results(
                beampowers=beampowers,
                settings=settings,
                start_time=start_time,
                grid_lon_coords=grid_lon_coords,
                grid_lat_coords=grid_lat_coords,
                station_locations=station_locations,
                plot_identifier=f"{fp[0]}_{fp[1]}_{wavetype}_{n_svd}_{noise_idx}",
            )

