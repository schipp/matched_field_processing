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

    from numpy import argmin, array
    from obspy import UTCDateTime
    from pandas import read_csv

    if settings["freq_band_mode"] == "timelist":
        timelist_df = read_csv(settings["external_timelist"])
        _times = timelist_df["time"].values
        fmin_per_starttime, fmax_per_starttime = (
            timelist_df["fmin"].values,
            timelist_df["fmax"].values,
        )
        _times = array([UTCDateTime(_) for _ in _times])

    # for start_time in start_times:
    #     for fp in settings["filterpairs"]:
    #         for wavetype in settings["wavetypes"]:
    for start_time, wavetype in product(start_times, settings["wavetypes"]):
        if settings["freq_band_mode"] == "timelist":
            fp = (
                fmin_per_starttime[argmin(abs(_times - start_time))],
                fmax_per_starttime[argmin(abs(_times - start_time))],
            )
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

        else:
            for fp in settings["frequency_bands"]:
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


def check_settings_valid(settings):
    """ Perform some sanity checks whether all settings have (proably) been set correctly.

    :param settings: dict containing all parameters for current run.
    :type settings: dict
    """

    import numbers

    from obspy import UTCDateTime
    from pandas import read_csv

    # 1 -- GLOBAL SETTINGS

    assert isinstance(
        settings["project_basedir"], str
    ), "project_basedir must be a string"

    assert isinstance(settings["project_id"], str), "project_id must be a string"

    # 2 -- TIME

    assert isinstance(
        settings["window_length"], int
    ), "window_length must be an integer"

    if settings["time_window_mode"] not in ["file", "overlapping_windows"]:
        raise ValueError('time_window_mode must be "file" or "overlapping_windows"')

    try:
        UTCDateTime(settings["start_time"])
    except:
        raise ValueError("start_time must be obspy.UTCDateTime-readable")

    try:
        UTCDateTime(settings["end_time"])
    except:
        raise ValueError("end_time must be obspy.UTCDateTime-readable")

    if (
        settings["time_window_mode"] == "overlapping_windows"
        and not 0 <= settings["overlap"] < 1
    ):
        raise ValueError("overlap must be 0 =< overlap < 1")

    if settings["time_window_mode"] == "file":
        try:
            df = read_csv(settings["external_timelist"])
            _ = [UTCDateTime(_) for _ in df["time"]]
        except:
            raise ValueError(
                "external_timelist must be a .csv file with a column 'time' containing obspy.UTCDateTime-readable times"
            )

    assert isinstance(
        settings["do_only_one_timewindow"], bool
    ), "do_only_one_timewindow must be true/false"

    # 3 -- GEOMETRY

    if settings["geometry_type"] not in ["cartesian", "geographic"]:
        raise ValueError("geometry_type must be 'cartesian' or 'geographic'")

    if settings["geometry_type"] == "cartesian":
        if not (
            len(settings["grid_limits_x"]) == 2
            and isinstance(settings["grid_limits_x"][0], numbers.Number)
            and isinstance(settings["grid_limits_x"][0], numbers.Number)
        ):
            raise ValueError(
                "grid_limits_x must be a list with two items, both must be numbers."
            )
        if not (
            len(settings["grid_limits_y"]) == 2
            and isinstance(settings["grid_limits_y"][0], numbers.Number)
            and isinstance(settings["grid_limits_y"][0], numbers.Number)
        ):
            raise ValueError(
                "grid_limits_y must be a list with two items, both must be numbers."
            )

    if settings["geometry_type"] == "geographic":
        # TODO: check positions are valid for Earth
        if not (
            len(settings["grid_limits_lon"]) == 2
            and isinstance(settings["grid_limits_lon"][0], numbers.Number)
            and isinstance(settings["grid_limits_lon"][0], numbers.Number)
        ):
            raise ValueError(
                "grid_limits_lon must be a list with two items, both must be numbers."
            )
        if not (
            len(settings["grid_limits_lat"]) == 2
            and isinstance(settings["grid_limits_lat"][0], numbers.Number)
            and isinstance(settings["grid_limits_lat"][0], numbers.Number)
        ):
            raise ValueError(
                "grid_limits_lat must be a list with two items, both must be numbers."
            )

    assert isinstance(
        settings["grid_spacing"], numbers.Number
    ), "grid_spacing must be a number."

    # 4 -- DATA

    assert isinstance(settings["data_fn"], list), "data_fn must be a list"
    assert all(
        [isinstance(_, str) for _ in settings["data_fn"]]
    ), "all items in 'data_fn' must be strings"

    assert isinstance(settings["sta_xml_dir"], str), "sta_xml_dir must be a string"

    assert isinstance(
        settings["sampling_rate"], numbers.Number
    ), "sampling_rate must be a number."

    # 5 -- SYNTHETICS

    assert isinstance(settings["do_synth"], bool), "do_synth must be true/false"

    if settings["do_synth"]:
        assert settings["synth_data_type"] in [
            "database_GF",
            "ricker",
        ], "synth_data_type must be 'database_GF' or 'ricker'."

        assert all(
            len(_) == 2 for _ in settings["synth_sources"]
        ), "synth_sources must be a list of 2-item lists for each source."

        assert (
            settings["add_noise_to_synth"] <= 0
        ), "add_noise_to_synth must be 0 or a positive number."

        assert isinstance(
            settings["add_noise_iterations"], int
        ), "add_noise_iterations must an integer."

        assert isinstance(
            settings["use_synth_stations"], bool
        ), "use_synth_stations must be true/false"

        if settings["use_synth_stations"]:
            if not settings["synth_stations_mode"] in [
                "file",
                "grid",
                "uniform",
                "partial_circle",
                "real_locations_worldwide",
            ]:
                raise ValueError(
                    "synth_stations_mode must be 'file', 'grid', 'uniform', 'partial_circle', or 'real_locations_worldwide'"
                )

            if settings["synth_stations_mode"] == "file":
                try:
                    df = read_csv(settings["synth_stations_file"])
                    _ = df["x"].values, df["y"].values

                except:
                    raise ValueError(
                        "synth_stations_file must be a .csv file with x,y columns"
                    )
            elif settings["synth_stations_mode"] == "file":
                assert isinstance(
                    settings["synth_stations_circle_n"], int
                ), "synth_stations_circle_n must be an integer."
                assert isinstance(
                    settings["synth_stations_circle_max"], int
                ), "synth_stations_circle_max must be an integer."
                assert (
                    settings["synth_stations_circle_max"]
                    >= settings["synth_stations_circle_n"]
                ), "synth_stations_circle_n must be an integer."
                assert isinstance(
                    settings["synth_stations_circle_radius"], numbers.Number
                ), "synth_stations_circle_radius must be a number."

            elif (
                settings["synth_stations_mode"] == "grid"
                or settings["synth_stations_mode"] == "uniform"
            ):
                assert isinstance(
                    settings["synth_stations_n"], int
                ), "synth_stations_n must be an integer"

            assert isinstance(settings["use_open_worldwide_stations"], bool)

    # 6 -- GREEN's FUNCTION DATABASE (instaseis)

    assert isinstance(settings["gf_db_dir"], str)

    # 7 -- SYNTHETIC WAVEFIELD TO MATCH AGAINST

    assert settings["type_of_gf"] in [
        "GF",
        "v_const",
    ], "type_of_gf must be 'GF' or 'v_const'"

    assert settings["source_depth"] >= 0

    assert settings["amplitude_treatment"] in [
        "None",
        "whitening_GF",
        "time_domain_norm",
        "surface_wave_spreading",
    ], "amplitude_treatment must be 'None', 'whitening_GF', 'time_domain_norm', 'surface_wave_spreading'"

    if settings["type_of_gf"]:
        assert settings["v_const"] > 0

    # 8 -- FREQUENCIES

    if settings["freq_band_mode"] == "global":
        assert all(
            True if fp is None else len(fp) == 2 for fp in settings["frequency_bands"]
        )
        assert all(
            True if fp is None else fp[0] < fp[1] for fp in settings["frequency_bands"]
        )
    elif settings["freq_band_mode"] == "timelist":
        timelist_df = read_csv(settings["external_timelist"])
        assert all(
            fmin < fmax
            for fmin, fmax in zip(
                timelist_df["fmin"].values, timelist_df["fmax"].values
            )
        )

    else:
        raise ValueError("Invalid option for 'freq_band_mode'")

    # 9 -- SOURCE MECHANISM

    assert isinstance(settings["strike_dip_rake_gridsearch"], bool)
    if not settings["strike_dip_rake_gridsearch"]:
        assert 0 < settings["strike_spacing"] <= 180
        assert 0 < settings["dip_spacing"] <= 90
        assert 0 < settings["rake_spacing"] <= 360
    else:
        assert len(settings["MT"]) == 6

    # 10 -- COMPUTATIONAL EFFICIENCY

    assert settings["n_processes"] > 1, "single-core NOT supported atm"

    assert isinstance(abs(settings["decimal_round"]), int)

    assert isinstance(settings["exclude_land"], bool)

    # 11 -- MISC

    assert isinstance(settings["do_plot"], bool)

    assert settings["components"] == ["Z"]
    assert settings["wavetypes"] == ["Z"]

    assert isinstance(settings["do_svd"], bool)
    if settings["do_svd"]:
        assert isinstance(settings["n_svd_components"], list)
        assert all(isinstance(_, int) for _ in settings["n_svd_components"])
