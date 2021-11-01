from numpy.core.fromnumeric import std
from numpy.random import standard_normal


def get_lorenzo_position_for_time(time):

    import numpy as np
    import obspy

    lorenzo_path = [
        (10.9, -18.8, 25),
        (11.0, -20.4, 30),
        (11.0, -21.9, 35),
        (11.1, -23.3, 40),
        (11.2, -24.6, 45),
        (11.4, -25.9, 50),
        (11.8, -27.2, 55),
        (12.2, -28.7, 55),
        (12.6, -30.2, 55),
        (13.0, -31.6, 60),
        (13.4, -33.0, 70),
        (13.9, -34.5, 75),
        (14.4, -36.0, 80),
        (14.5, -37.5, 85),
        (14.7, -38.8, 95),
        (15.2, -39.8, 105),
        (16.0, -40.6, 115),
        (17.1, -41.2, 125),
        (18.1, -41.9, 125),
        (18.9, -42.7, 120),
        (19.6, -43.5, 110),
        (20.3, -44.2, 105),
        (21.1, -44.6, 100),
        (22.0, -44.9, 105),
        (22.9, -45.0, 115),
        (23.8, -45.0, 130),
        (24.3, -45.0, 140),
        (24.7, -45.0, 130),
        (25.6, -44.8, 110),
        (26.4, -44.4, 90),
        (27.2, -44.0, 90),
        (28.2, -43.6, 90),
        (29.2, -43.2, 90),
        (30.2, -42.7, 90),
        (31.2, -41.9, 90),
        (32.6, -40.7, 85),
        (34.3, -39.0, 85),
        (35.9, -36.8, 85),
        (37.8, -34.4, 85),
        (40.2, -31.4, 80),
        (43.0, -28.0, 70),
        (45.9, -24.4, 65),
        (49.2, -21.5, 60),
        (52.0, -18.7, 60),
        (54.5, -15.7, 60),
        (55.8, -13.3, 60),
        (55.4, -10.5, 55),
        (54.5, -8.2, 50),
    ]

    lorenzo_times = np.array(
        [
            obspy.UTCDateTime("2019-09-22Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-23Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-23Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-23Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-23Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-24Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-24Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-24Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-24Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-25Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-25Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-25Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-25Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-26Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-26Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-26Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-26Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-27Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-27Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-27Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-27Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-28Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-28Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-28Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-28Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-29Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-29Z03:00:00.0Z"),
            obspy.UTCDateTime("2019-09-29Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-29Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-29Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-09-30Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-09-30Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-09-30Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-09-30Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-10-01Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-10-01Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-10-01Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-10-01Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-10-02Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-10-02Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-10-02Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-10-02Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-10-03Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-10-03Z06:00:00.0Z"),
            obspy.UTCDateTime("2019-10-03Z12:00:00.0Z"),
            obspy.UTCDateTime("2019-10-03Z18:00:00.0Z"),
            obspy.UTCDateTime("2019-10-04Z00:00:00.0Z"),
            obspy.UTCDateTime("2019-10-04Z06:00:00.0Z"),
        ]
    )

    return lorenzo_path[np.argmin(np.abs(lorenzo_times - time))]


def chunks(lst, n):
    """Yield n chunks and their indices from lst."""
    import numpy as np

    # TODO: optmize chunks to evenly distribute land cells

    chunk_size = len(lst) // n
    if len(lst) % 2 != 0:
        chunk_size += 1
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i : i + chunk_size]
        chunk_indices = np.arange(i, i + chunk_size)
        yield chunk_indices, chunk


def settings_gen(start_times, settings):
    """Generator that yields parameter combinations to loop over.

    :param start_times: Start times MFP is to be computed for.
    :type start_times: list[obspy.UTCDateTime()]
    :param settings: Dataclass holding all global parameters for current run.
    :type settings: Parameters()
    :yield: settings for current start_time, filter-pair, wavetyipe, number of SVD components, noise iteration
    :rtype: tuple or list
    """
    from itertools import product

    # for start_time in start_times:
    #     for fp in settings["filterpairs"]:
    #         for wavetype in settings["wavetypes"]:
    for start_time, fp, wavetype in product(
        start_times, settings["filterpairs"], settings["wavetypes"]
    ):
        if settings["do_svd"]:
            for n_svd in settings["n_svd_components"]:
                if settings["add_noise_to_synth"] > 0:
                    for noise_idx in range(settings["add_noise_iterations"]):
                        if settings["strike_dip_rake_gridsearch"]:
                            for sdr in sdr_grid_gen(settings):
                                yield start_time, fp, wavetype, n_svd, noise_idx, sdr
                        else:
                            yield start_time, fp, wavetype, n_svd, noise_idx, False
                else:
                    if settings["strike_dip_rake_gridsearch"]:
                        for sdr in sdr_grid_gen(settings):
                            yield start_time, fp, wavetype, n_svd, 0, sdr
                    else:
                        yield start_time, fp, wavetype, n_svd, 0, False
        else:
            if settings["add_noise_to_synth"] > 0:
                for noise_idx in range(settings["add_noise_iterations"]):
                    if settings["strike_dip_rake_gridsearch"]:
                        for sdr in sdr_grid_gen(settings):
                            yield start_time, fp, wavetype, 0, noise_idx, sdr
                    else:
                        yield start_time, fp, wavetype, 0, noise_idx, False
            else:
                if settings["strike_dip_rake_gridsearch"]:
                    for sdr in sdr_grid_gen(settings):
                        yield start_time, fp, wavetype, 0, 0, sdr
                else:
                    yield start_time, fp, wavetype, 0, 0, False


def sdr_grid_gen(settings):
    from itertools import product

    import numpy as np

    strikes = np.arange(180, 361, settings["strike_spacing"])
    dips = np.arange(0, 91, settings["dip_spacing"])
    rakes = np.arange(-180, 181, settings["rake_spacing"])
    for sdr in product(strikes, dips, rakes):
        yield sdr


def whiten_spectrum(spec):
    import numpy as np

    # compute energy
    E = np.sqrt(np.sum(np.absolute(np.square(spec)))) / (len(spec))
    # take only angles (phase) and multiply by normalized energy
    spec = np.exp(1j * np.angle(spec)) * E
    return spec
