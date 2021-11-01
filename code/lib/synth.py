# from .geometry import get_distances
import numpy as np
from scipy.signal.filter_design import freqs
from tqdm import tqdm

from .gf import get_gf_spectrum


def get_synth_stations(settings, wiggle=0.5):
    from itertools import product

    import numpy as np

    mode = settings["synth_stations_mode"]
    n = settings["synth_stations_n"]

    if mode == "grid":
        lons = np.linspace(-180, 180 - (360 / int(np.sqrt(n))), int(np.sqrt(n)))
        lats = np.linspace(-75, 75, int(np.sqrt(n)))
        station_locations = list(product(lons, lats))

    elif mode == "uniform":
        lons = np.random.uniform(low=0, high=180, size=n)
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

    # if wiggle != 0:
    # station_locations = [[sta_lon+np.random.uniform(-wiggle, wiggle), sta_lat+np.random.uniform(-wiggle, wiggle)] for sta_lon, sta_lat in product(lons, lats)]

    station_locations = np.array(station_locations)

    return station_locations


def get_real_station_locations_worldwide():
    import numpy as np
    import obspy
    from obspy.clients.fdsn import Client

    client = Client("IRIS")

    inv = client.get_stations(
        starttime=obspy.UTCDateTime("2019-09-25T00:00:00.0Z"),
        endtime=obspy.UTCDateTime("2019-10-05T00:00:00.0Z"),
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
    import numpy as np

    # xn(t) = cos(2𝜋f(tn + Δt))
    # times = np.arange(0, settings['window_length'] / settings['sampling_rate'], 1 / settings['sampling_rate'])
    # trace = np.cos(2 * np.pi * f * (traveltime + times))
    from scipy.signal import ricker

    time = dist / settings["v_const"]
    # delta-response -> simple travel time
    medium_response = np.zeros(settings["window_length"])
    medium_response[int(time * settings["sampling_rate"])] = 1

    wavelet = ricker(5 * settings["sampling_rate"], a=1)

    seismogram = np.convolve(wavelet, medium_response, mode="same")
    # seismogram /= np.max(np.abs(seismogram))

    if settings["add_noise_to_synth"] > 0:
        noise_bounds = settings["add_noise_to_synth"] * np.max(np.abs(seismogram))
        seismogram += np.random.uniform(-noise_bounds, noise_bounds, seismogram.shape,)

    return seismogram


def get_distances(settings, list_of_locs: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Compute the distance between an array of coordinate-pairs and a single point.
    If you want 3D distances (source at depth) list_of_locs = [(lat1, lon1, alt1), ...]
    """
    # cartesian
    if settings["geometry_type"] == "cartesian":
        dists = np.linalg.norm(list_of_locs - point, ord=2, axis=1)
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
    dists_azs[:, 0] /= 1000  # to km

    return np.array(dists_azs)


def get_synth_spectra(
    station_locations, settings, instaseis_db, freqs_of_interest_idx=None,
):
    import numpy as np

    # compute beampowers
    synth_sources_spectra = []
    for synth_source in settings["synth_sources"]:
        dists_azs_to_src = get_distances(
            settings, list_of_locs=station_locations, point=synth_source
        )

        # if settings.components == ['Z']:
        #     dists_azs_to_src[:, :, 1] = 0

        # synth_source_spectra = np.array([get_gf_spectrum(freqs, dist, vel, instaseis_db=db)[freqs_of_interest_idx] for dist in tqdm(dists_to_src)])

        synth_source_spectra = []
        for dist_az in tqdm(dists_azs_to_src, desc="computing synthetic data"):
            if settings["type_of_gf"] == "GF":
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

            elif settings["type_of_gf"] == "v_const":
                dist = dist_az[0]
                synth_trace = get_synth_wave(dist, settings)
                from scipy.fft import fft

                tr_spectrum = fft(synth_trace)
                # spec_freqs = fftfreq(len(synth_trace), settings['sampling_rate'])
                # spec_freqs_of_interest_idx = (spec_freqs >= fp[0]) & (spec_freqs <= fp[1])
                # normalize for each station individually
                # data_spectrum = tr_spectrum[spec_freqs_of_interest_idx]
                if freqs_of_interest_idx is None:
                    data_spectrum = tr_spectrum
                else:
                    data_spectrum = tr_spectrum[freqs_of_interest_idx]
                # /np.max(np.abs(tr_spectrum[spec_freqs_of_interest_idx]))
                synth_source_spectra.append(data_spectrum)
        synth_sources_spectra.append(np.array(synth_source_spectra))

    synth_sources_spectra = np.array(synth_sources_spectra)
    return np.sum(synth_sources_spectra, axis=0)


def get_data_spectra(st, start_time, fp, settings):

    # compute spectra for all data
    from scipy.fftpack import fft, fftfreq

    data_spectra = []
    # replace trim by time window based on arrival
    for tr in st_curr:
        # trim tr to given time window
        # tr_start = tr.stats.starttime
        # idx_start = tr.stats.samplingrate * (tr_start + distance / vel - window_length_around_rayleigh_arrival)
        # idx_end = tr.stats.samplingrate * (tr_start + distance / vel + window_length_around_rayleigh_arrival)
        # data = tr.data[idx_start:idx_end]

        tr_spectrum = fft(tr.data[:-1])
        spec_freqs = fftfreq(len(tr.data[:-1]), settings["sampling_rate"])
        spec_freqs_of_interest_idx = (spec_freqs >= fp[0]) & (spec_freqs <= fp[1])
        # normalize for each station individually
        # data_spectrum = tr_spectrum[spec_freqs_of_interest_idx]
        data_spectrum = tr_spectrum[spec_freqs_of_interest_idx]
        # /np.max(np.abs(tr_spectrum[spec_freqs_of_interest_idx]))
        data_spectra.append(data_spectrum)

    return np.array(data_spectra)
