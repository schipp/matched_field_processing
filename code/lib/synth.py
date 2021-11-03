import numpy as np
from tqdm import tqdm

from .gf import get_gf_spectrum


def get_synth_stations(settings, wiggle=0):
    """ Compute synthetic station locations. 
    
    Values for mode "grid" and "uniform" and currently for tests on global Earth geometry.
    TODO: incorporate into settings.yml

    :param settings: dict holding all info for project
    :type settings: dict
    :param wiggle: adds random variations in interval [-wiggle, wiggle] to locations, defaults to 0
    :type wiggle: float, optional
    :return: array containing station locations longitude/x and latitude/y coordinates. shape = (n, 2)
    :rtype: numpy.ndarray
    """
    from itertools import product

    mode = settings["synth_stations_mode"]
    n = settings["synth_stations_n"]

    if mode == "grid":
        lons = np.linspace(-180, 180 - (360 / int(np.sqrt(n))), int(np.sqrt(n)))
        lats = np.linspace(-75, 75, int(np.sqrt(n)))
        station_locations = list(product(lons, lats))

    elif mode == "uniform":
        lons = np.random.uniform(low=-180, high=180, size=n)
        lats = np.random.uniform(low=-75, high=75, size=n)
        station_locations = list(zip(lons, lats))

    elif mode == "partial_circle":
        n_total = settings["synth_stations_circle_max"]
        radius = settings["synth_stations_circle_radius"]
        n_used = settings["synth_stations_circle_n"]

        azimuths = np.linspace(0, 2 * np.pi, n_total)
        azimuths_used = azimuths[:n_used]

        lons = radius * np.cos(azimuths_used)
        lats = radius * np.sin(azimuths_used)
        station_locations = list(zip(lons, lats))

    elif mode == "file":
        import pandas as pd

        df = pd.read_csv(settings["synth_stations_file"])
        lons = df["x"].values
        lats = df["y"].values
        station_locations = list(zip(lons, lats))

    if wiggle != 0:
        station_locations = [
            [
                sta_lon + np.random.uniform(-wiggle, wiggle),
                sta_lat + np.random.uniform(-wiggle, wiggle),
            ]
            for sta_lon, sta_lat in product(lons, lats)
        ]

    station_locations = np.array(station_locations)

    return station_locations


def get_real_station_locations_worldwide():
    """Get worldwide station locations from IRIS.

    :return: array containing station locations longitude and latitude coordinates. shape = (n, 2)
    :rtype: numpy.ndarray
    """

    import numpy as np
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client

    client = Client("IRIS")

    inv = client.get_stations(
        starttime=UTCDateTime("2019-09-25T00:00:00.0Z"),
        endtime=UTCDateTime("2019-10-05T00:00:00.0Z"),
        channel="*HZ",
        includerestricted=False,
        level="channel",
    )

    station_locations = []
    for net in inv:
        for sta in net:
            lat, lon = sta.latitude, sta.longitude
            station_locations.append([lon, lat])

    return np.array(station_locations)


def get_synth_wave(dist, settings):
    """ Computes a ricker wavelet recorded after distance dist.

    :param dist: Distance that the wave has travelled, uses velocity settings["v_const"].
    :type dist: float
    :param settings: dict holding all info for project
    :type settings: dict
    :return: Synthetic seismogram recorded at distance dist.
    :rtype: np.ndarray
    """
    import numpy as np
    from scipy.signal import ricker

    time = dist / settings["v_const"]
    # delta-response -> simple travel time
    medium_response = np.zeros(settings["window_length"])
    medium_response[int(time * settings["sampling_rate"])] = 1

    wavelet = ricker(5 * settings["sampling_rate"], a=1)

    seismogram = np.convolve(wavelet, medium_response, mode="same")

    if settings["add_noise_to_synth"] > 0:
        noise_bounds = settings["add_noise_to_synth"] * np.max(np.abs(seismogram))
        seismogram += np.random.uniform(-noise_bounds, noise_bounds, seismogram.shape,)

    return seismogram


def get_distances(settings, list_of_locs: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Compute the distance between an array of coordinate-pairs and a single point.

    :param settings: dict holding all info for project
    :type settings: dict
    :param list_of_locs: the locations of all points for which the distance to point is computed
    :type list_of_locs: np.ndarray
    :param point: the reference point that distances are computed to
    :type point: np.ndarray
    :return: distances and azimuths for each list_of_locs to point combination
    :rtype: np.ndarray
    """

    # cartesian
    if settings["geometry_type"] == "cartesian":
        dists = np.linalg.norm(list_of_locs - point, ord=2, axis=1)
        # Ignore azimuths for now, only concerned with vertical explosion data (az irrelevant).
        # TODO: compute correct azimuths
        azs = np.zeros(len(dists))
        return np.array(list(zip(dists, azs)))

    # geometric
    from obspy.geodetics import gps2dist_azimuth

    dists_azs = [
        gps2dist_azimuth(
            lat1=sta_loc[1], lon1=sta_loc[0], lat2=point[1], lon2=point[0]
        )[:2]
        for sta_loc in list_of_locs
    ]

    dists_azs = np.array(dists_azs)
    # convert distances from m to km
    dists_azs[:, 0] /= 1000

    return np.array(dists_azs)


def get_synth_spectra(
    station_locations, settings, instaseis_db, freqs_of_interest_idx=None,
):
    """ Compute synthetic spectra for synthetic tests.

    :param station_locations: locations of stations for which to compute synthetic spectra for
    :type station_locations: np.ndarray
    :param settings: dict holding all info for project
    :type settings: dict
    :param instaseis_db: open connection to an instaseis db
    :type instaseis_db: instaseis.InstaseisDB
    :param freqs_of_interest_idx: indices of frequencies to limit to, defaults to None
    :type freqs_of_interest_idx: np.ndarray, optional
    :return: the synthetic spectra expected on each station in station_locations
    :rtype: np.ndarray
    """

    synth_sources_spectra = []
    for idx, synth_source in enumerate(settings["synth_sources"]):
        dists_azs_to_src = get_distances(
            settings, list_of_locs=station_locations, point=synth_source
        )

        synth_source_spectra = []
        for dist_az in tqdm(
            dists_azs_to_src,
            desc=f"computing synthetic data {idx}/{len(settings['synth_sources'])}",
            leave=False,
            # disable=~settings["verbose"],
        ):
            if settings["synth_data_type"] == "database_GF":
                spec_z, spec_r, spec_t = get_gf_spectrum(
                    dist_az, settings, instaseis_db=instaseis_db, is_synth=True
                )
                if freqs_of_interest_idx is not None:
                    spec_z = spec_z[freqs_of_interest_idx]
                    spec_r = spec_r[freqs_of_interest_idx]
                    spec_t = spec_t[freqs_of_interest_idx]
                if "Z" in settings["components"]:
                    synth_source_spectra.append(spec_z)
                if "R" in settings["components"]:
                    synth_source_spectra.append(spec_r)
                if "T" in settings["components"]:
                    synth_source_spectra.append(spec_t)

            elif settings["synth_data_type"] == "ricker":
                dist = dist_az[0]
                synth_trace = get_synth_wave(dist, settings)
                from scipy.fft import fft

                tr_spectrum = fft(synth_trace)

                if freqs_of_interest_idx is None:
                    data_spectrum = tr_spectrum
                else:
                    data_spectrum = tr_spectrum[freqs_of_interest_idx]
                synth_source_spectra.append(data_spectrum)
        synth_sources_spectra.append(np.array(synth_source_spectra))

    # sum over individual sources
    synth_sources_spectra = np.sum(np.array(synth_sources_spectra), axis=0)
    return synth_sources_spectra
