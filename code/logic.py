# preamble
import logging
import os
import time

import numpy as np
import obspy
import yaml
from tqdm import tqdm

from lib.beam import get_beampowers
from lib.data import get_data_spectra, get_data_traces
from lib.geometry import (
    generate_global_grid,
    get_all_distances_rounded,
    get_loc_from_timelist,
    get_station_locations,
)
from lib.gf import get_gf_spectra_for_dists, get_gf_spectrum, get_gf_traces_for_dists
from lib.math import get_csdm, svd_csdm, weigh_spectra_by_coherency
from lib.misc import get_lorenzo_position_for_time, settings_gen, whiten_spectrum
from lib.plotting import plot_beampowers_on_map
from lib.synth import get_synth_spectra
from lib.time import get_start_times

if __name__ == "__main__":
    # load settings
    run_time = time.time()

    with open("settings.yml", "r") as stream:
        settings = yaml.safe_load(stream)

    # project dirs and create if necessary
    project_dir = f"../projects/{settings['project_id']}/"
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
    from shutil import copyfile

    copyfile(f"settings.yml", f"{project_dir}/{run_time}_settings.yml")
    logging.info(f"Copied settings.yml into {project_dir}/{run_time}_settings.yml")

    # define time windows
    start_times = np.array(get_start_times(settings))
    # HACK to only do first time window
    if settings["do_only_one_timewindow"]:
        start_times = start_times[:1].tolist()
        start_times.append([])

    logging.info(f"{start_times=}")

    # load data
    if settings["do_synth"] and settings["use_synth_stations"]:
        st = None
    else:
        st = obspy.Stream()
        for data_fn in settings["data_fn"]:
            st += obspy.read(data_fn)
        logging.info("Data after reading:")
        logging.info(st)

        st.merge(fill_value=0)

        st_tmp = obspy.Stream()
        if settings["do_quality_check"]:
            logging.info(f"Starting quality check")
            amps = np.array([np.max(np.abs(tr.data)) for tr in st])
            stds = np.array([np.std(tr.data) for tr in st])
            # nonzero_ratios = [np.count_nonzero(tr.data)/len(tr.data) for tr in st]
            # energies = [np.linalg.norm(tr.data, ord=2) for tr in st]
            # testing
            amps_good = (amps > settings["amp_thresholds"][0]) & (
                amps < settings["amp_thresholds"][1]
            )
            std_good = (stds > settings["std_thresholds"][0]) & (
                stds < settings["std_thresholds"][1]
            )
            for t_idx, tr in enumerate(st):
                if not t_idx in np.where((amps_good) & (std_good))[0]:
                    logging.info(f"Skipping {tr.id}")
                    continue
                st_tmp += tr
            st = st_tmp
            logging.info("Data after quality check:")
            logging.info(st)

        # remove hardcoded outliers
        # print(len(st))
        # if not tr.stats.station in ['WB22', 'TRI', 'MUD', 'NRS', 'POPM', 'CORL', 'FVI', 'MCSR', 'MPAZ', 'PARC']:
        logging.info(f"Manual station removal:")
        st_tmp = obspy.Stream()
        for tr in st:
            # if not tr.stats.station in ['KESW', 'ESK', 'EDI']:
            #     st_tmp += tr
            if not tr.stats.network in []:
                st_tmp += tr
            else:
                logging.info(f"Skipping {tr.id}")
        st = st_tmp
        logging.info("Data after manual station removal:")
        logging.info(st)

        if settings["normalize_data"]:
            if settings["norm_mode_data"] == "whitening":
                for tr in st:
                    spec = np.fft.fft(tr.data)
                    spec = whiten_spectrum(spec)
                    tr.data = np.fft.ifft(spec).real
            else:
                st.normalize()

        # print(len(st))
        # st.resample(settings['sampling_rate'])

        logging.info(f"Checking duplicate stations...")
        # check for duplicate stations
        station_names = [tr.stats.station for tr in st]
        unique_stations, unique_counts = np.unique(station_names, return_counts=True)
        duplicate_stations = unique_stations[unique_counts > 1]
        # keep only one of duplicate entries
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
        for start_time in tqdm(start_times, desc="checking time windows"):
            if isinstance(start_time, list):
                continue
            # st_curr = st.copy()
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
                # logging.info(f"Data removed for starttime {start_time}: {removed_traces}")
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

    # generate grid-points
    grid_points, grid_lon_coords, grid_lat_coords = generate_global_grid(
        settings=settings
    )

    logging.info(f"{grid_points.shape=}")

    # check if results already exist
    # for idx, start_time in tqdm(enumerate(start_times), total=len(start_times), desc='time windows'):
    # for idx, (start_time, fp) in tqdm(enumerate(zip(start_times, settings['filterpairs'])), total=len(start_times), desc='Checking Re-plotting'):
    one_missing = False
    settings_generator = settings_gen(start_times, settings)
    for start_time, fp, wavetype, n_svd, noise_idx, sdr in settings_generator:
        if isinstance(start_time, list):
            continue
        filename = f"{project_dir}/out/out_{start_time.timestamp}_{settings['window_length']}_{fp}_{n_svd}_{noise_idx}_{sdr}.npy"
        if os.path.isfile(f"{filename}"):
            logging.warning(f"Output {filename} already exists. Re-plotting.")
            if settings["do_plot"]:
                beampowers = np.load(f"{filename}", allow_pickle=True)
                station_locations = get_station_locations(settings, st)
                # # stacked beampowers over all timewindows
                # if len(beampowers_per_start_time) > 1:
                #     beampower_stack = np.mean(beampowers_per_start_time, axis=0)
                #     plot_beampowers_on_map(
                #         lons=grid_lon_coords,
                #         lats=grid_lat_coords,
                #         beampowers=beampower_stack,
                #         station_locations=station_locations,
                #         outfile=f'{project_dir}/plots/stack.png')

                # if len(beampowers_per_start_time) == 1:
                #     beampowers_per_start_time.append([])

                # beampowers over individual timewindows
                # for idx, beampowers in enumerate(beampowers_per_start_time):
                # if len(beampowers) == 0:
                #     break

                beampowers_to_plot = beampowers / np.nanmax(np.abs(beampowers))

                if settings["do_synth"]:
                    source_loc = np.array(settings["synth_sources"]).T
                else:
                    lat, lon, windspeed = get_lorenzo_position_for_time(start_time)
                    source_loc = lon, lat
                # CHILE EVENT 2019-09-29 15:57:56.0 UTC
                lat, lon = -35.47, -72.91
                # CHINO HILLS
                lat, lon = 33.953, -117.761
                lon, lat = get_loc_from_timelist(start_time, settings)
                source_loc = lon, lat

                if settings["do_synth"]:
                    source_loc = np.array(settings["synth_sources"]).T

                plot_beampowers_on_map(
                    lons=grid_lon_coords,
                    lats=grid_lat_coords,
                    beampowers=beampowers_to_plot,
                    station_locations=station_locations,
                    settings=settings,
                    outfile=f"../projects/{settings['project_id']}/plots/{start_time.timestamp}_{fp[0]}_{fp[1]}_{wavetype}_{n_svd}_{noise_idx}.png",
                    source_loc=source_loc,
                    title=start_time,
                    plot_station_locations=True,
                )
        else:
            one_missing = True
    if not one_missing:
        import sys

        sys.exit("All maps re-plotted. Exiting")

    logging.info("At least one start_time missing - proceeding")

    station_locations = get_station_locations(settings, st)
    logging.info(f"{station_locations.shape=}")

    # define indices of frequency range
    from scipy.fftpack import fftfreq

    n_samples = settings["window_length"] * settings["sampling_rate"]
    freqs = fftfreq(n_samples, settings["sampling_rate"])

    # remove grid points on land?
    # how to handle meshgrid / plotting properly?
    # create a trigger in bp.py to "skip" gps that are onland (i.e., set them = nan)
    if settings["do_synth"]:
        gp_on_land = np.zeros(len(grid_points), dtype=bool)
        if settings["geometry_type"] == "geographic":
            from global_land_mask import globe

            gp_on_land = globe.is_land(grid_points[:, 1], grid_points[:, 0])
    else:
        from global_land_mask import globe

        gp_on_land = globe.is_land(grid_points[:, 1], grid_points[:, 0])
    # gp_on_land = [globe.is_land(gp[1], gp[0]) for gp in grid_points]
    # grid_points = np.array([gp for gp in grid_points if globe.is_ocean(gp[1], gp[0]])

    # compute all distance-azimuth-combinations of gridpoints-station geometry
    gp_dists = get_all_distances_rounded(
        station_locations=station_locations, grid_points=grid_points, settings=settings,
    )
    # logging.info(f"{np.max(gp_dists, axis=0)=}")

    logging.info(f"{gp_dists.shape=}")

    # azimuths irrelevant if only Z component, then set all to 0
    # can reduce cost massively
    # only applies if MT used is explosion
    if settings["components"] == ["Z"] and settings["MT"] == [1, 1, 1, 0, 0, 0]:
        gp_dists[:, :, 1] = 0

    # deal with distance == 0, because singularity
    # currently fix by adding half a bin (e.g., 50km for decimal_round = -2)
    # where_is_0 = np.where(gp_dists[:, :, 0] == 0)
    # print(gp_dists[where_is_0[0]])
    # print(gp_dists.shape)
    # gp_dists[
    #     np.where(gp_dists[:, :, 0] < (10 ** -settings["decimal_round"]))
    # ] == np.array([(10 ** -settings["decimal_round"]), 0])
    # gp_dists[np.where(gp_dists[:, :, 0] < 1)] = np.array([1, 0])
    # print(gp_dists.shape)
    # print(gp_dists[np.where(gp_dists[:, :, 0] == 0)].shape)
    #

    # fix zero distance / singularity
    # print("lets go")
    # azs_tmp = gp_dists[gp_dists[:, :, 0] < 2][:, 1]
    # # azs_tmp = gp_dists[gp_dists[:, :, 0] < (10 ** -settings["decimal_round"]) / 2][:, 1]
    # for az_tmp in tqdm(azs_tmp):
    #     gp_dists[
    #         np.where((gp_dists[:, :, 0] < 2) & (gp_dists[:, :, 1] == az_tmp))
    #     ] = np.array([2, az_tmp])

    # extract relevant distances and azimuths only
    relevant_dists_azs = np.unique(
        gp_dists.reshape(len(station_locations) * len(grid_points), 2), axis=0
    )

    logging.info(f"{relevant_dists_azs.shape=}")
    # deal wth distance == 0, because singularity issue
    # currently fix by adding half a bin (e.g., 50km for decimal_round = -2)
    # i = 0
    # while True:
    #     if relevant_dists_azs[i, 0] == 0:
    #         # move half a bin up
    #         relevant_dists_azs[i, 0] = (10 ** abs(settings.decimal_round))/2
    #         i += 1
    #     else:
    #         break
    # azs_tmp = gp_dists[gp_dists[:, :, 0] == 0][:, 1]
    # for az_tmp in azs_tmp:
    #     gp_dists[np.where((gp_dists[:, :, 0] == 0) & (gp_dists[:, :, 1] == az_tmp))] = np.array([(10 ** abs(settings.# decimal_round))/2, az_tmp])

    # intialize green's function database
    import instaseis

    instaseis_db = instaseis.open_db(settings["gf_db_dir"])
    logging.info(f"Initialized Instaseis DB at {settings['gf_db_dir']}")

    # compute gf spectra for rounded, non-duplicate, and non-zero gridpoint-station-distances and -azimuths
    gf_spectra_all = get_gf_spectra_for_dists(
        freqs=freqs,
        dists_azs=relevant_dists_azs,
        instaseis_db=instaseis_db,
        settings=settings,
    )

    # gf_traces_all = get_gf_traces_for_dists(
    #         freqs=freqs,
    #         dists_azs=relevant_dists_azs,
    #         instaseis_db=instaseis_db,
    #         settings=settings
    #         )

    logging.info(f"{gf_spectra_all.shape=}")

    settings_generator = settings_gen(start_times, settings)
    # run MFP
    beampowers_per_start_time = []
    # for idx, start_time in tqdm(enumerate(start_times), total=len(start_times), desc='time windows'):
    for start_time, fp, wavetype, n_svd, noise_idx, sdr in tqdm(settings_generator):
        logging.info(f"Starting: {start_time} - {fp}Hz - {wavetype}")

        # if grid-searching sdr, need to recompute GF each time
        if sdr:
            gf_spectra_all = get_gf_spectra_for_dists(
                freqs=freqs,
                dists_azs=relevant_dists_azs,
                instaseis_db=instaseis_db,
                settings=settings,
                sdr=sdr,
            )

        if isinstance(fp, list):
            freqs_of_interest_idx = (freqs >= fp[0]) & (freqs <= fp[1])
            # limit gf spectra to frequencies of interest
            gf_spectra = gf_spectra_all[:, freqs_of_interest_idx]
        else:
            freqs_of_interest_idx = None
            gf_spectra = gf_spectra_all
        logging.info(f"After limiting to relevant freqs: {gf_spectra.shape=}")

        # HACK to skip empty entry when doing only first start time (see above)
        if isinstance(start_time, list):
            continue

        filename = f"{project_dir}/out/out_{start_time.timestamp}_{settings['window_length']}_{fp}_{n_svd}_{noise_idx}_{sdr}.npy"
        if os.path.isfile(filename):
            logging.warning(f"output file {filename} exists - skipping")
            continue

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
            # data_traces = get_data_traces(st, start_time, fp, settings)

        # # filter gf_traces_all
        # gf_traces_filtered = []
        # for trace in gf_traces_all:
        #     _tr = obspy.Trace()
        #     _tr.data = trace
        #     _tr.stats.sampling_rate = 1
        #     _tr.filter("bandpass", freqmin=fp[0], freqmax=fp[1])
        #     gf_traces_filtered.append(_tr.data)
        # gf_traces_filtered = np.array(gf_traces_filtered)

        logging.info(f"{data_spectra.shape=}")

        # ensure spectra shape is same
        if data_spectra.shape[1] > gf_spectra.shape[1]:
            logging.critical(
                f"data_spectra too long: {data_spectra.shape[1]=} vs. {gf_spectra.shape[1]=}. SHOULD NOT HAPPEN"
            )
        if data_spectra.shape[1] < gf_spectra.shape[1]:
            logging.warning(
                f"data_spectra too short: {data_spectra.shape[1]=} vs. {gf_spectra.shape[1]=}. Removing highest {gf_spectra.shape[1] - data_spectra.shape[1]} frequencies."
            )
            gf_spectra = gf_spectra[:, : data_spectra.shape[1]]

        logging.info(f"{gf_spectra.shape=}")

        if settings["amplitude_treatment"] == "phase_correlation":
            gf_spectra = np.exp(1j * np.angle(gf_spectra))
            data_spectra = np.exp(1j * np.angle(data_spectra))

        # weighting by coherency
        if settings["do_coherency_weighting"]:
            logging.info(f"START: Coherency weighting ")
            data_spectra = weigh_spectra_by_coherency(data_spectra)
            # data_spectra[0, :] *= 1000
            logging.info(f"END: Coherency weighting")

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
            # data_traces=data_traces,
            # gf_traces=gf_traces_filtered,
            gp_dists=gp_dists,
            gp_on_land=gp_on_land,
            relevant_dists=relevant_dists_azs,
            settings=settings,
        )

        out_array = np.array(beampowers).reshape(
            len(grid_lon_coords), len(grid_lat_coords)
        )

        # beampowers_per_start_time.append(out_array)
        # save results
        # filename = f"{project_dir}/out/out_{start_time.timestamp}_{settings['window_length']}.npy"
        np.save(filename, out_array)

        # np.save(f'{project_dir}/out/geometry.npy', [station_locations])

        # plotting
        if settings["do_plot"]:
            # # stacked beampowers over all timewindows
            # if len(beampowers_per_start_time) > 1:
            #     beampower_stack = np.mean(beampowers_per_start_time, axis=0)

            #     plot_beampowers_on_map(
            #         lons=grid_lon_coords,
            #         lats=grid_lat_coords,
            #         beampowers=beampower_stack,
            #         station_locations=station_locations,
            #         outfile=f'{project_dir}/plots/stack.png')

            # if len(beampowers_per_start_time) == 1:
            #     beampowers_per_start_time.append([])

            # beampowers over individual timewindows
            # for idx, beampowers in enumerate(beampowers_per_start_time):
            # if len(beampowers) == 0:
            #     break

            beampowers_to_plot = out_array / np.nanmax(np.abs(out_array))

            if settings["do_synth"]:
                source_loc = np.array(settings["synth_sources"]).T
            else:
                lat, lon, windspeed = get_lorenzo_position_for_time(start_time)
                # CHINO HILLS
                lon, lat = get_loc_from_timelist(
                    start_time, settings
                )  # 33.953, -117.761
                source_loc = lon, lat

            plot_beampowers_on_map(
                lons=grid_lon_coords,
                lats=grid_lat_coords,
                beampowers=beampowers_to_plot,
                station_locations=station_locations,
                settings=settings,
                outfile=f"../projects/{settings['project_id']}/plots/{start_time.timestamp}_{fp[0]}_{fp[1]}_{wavetype}_{n_svd}_{noise_idx}.png",
                source_loc=source_loc,
                title=start_time,
                plot_station_locations=True,
            )

