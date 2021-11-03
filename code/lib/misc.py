def chunks(lst, n):
    """Divides lst into n chunks and yields a chunk."""
    import numpy as np

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
    :type start_times: list[obspy.UTCDateTime]
    :param settings: dict holding all global parameters for current run.
    :type settings: dict
    :yield: settings for current start_time, filter-pair, wavetyipe, number of SVD components, noise iteration
    :rtype: tuple or list
    """
    from itertools import product

    # for start_time in start_times:
    #     for fp in settings["filterpairs"]:
    #         for wavetype in settings["wavetypes"]:
    for start_time, fp, wavetype in product(
        start_times, settings["frequency_bands"], settings["wavetypes"]
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
    """Generator for searching strike, dip, and rake.
    Used when testing MFP for different source mechanisms.

    :param settings: [description]
    :type settings: [type]
    :yield: current strike, dip, and rake
    :rtype: list[float]
    """
    from itertools import product

    import numpy as np

    strikes = np.arange(180, 361, settings["strike_spacing"])
    dips = np.arange(0, 91, settings["dip_spacing"])
    rakes = np.arange(-180, 181, settings["rake_spacing"])
    for sdr in product(strikes, dips, rakes):
        yield sdr
