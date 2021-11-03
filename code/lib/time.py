def get_start_times(settings):
    """ Gets the correct start times depending on what's defined in settings

    :param settings: dict holding all info for project
    :type settings: dict
    :return: list of start times
    :rtype: list[obspy.UTCDateTime()]
    """
    if settings["time_window_mode"] == "file":
        return get_start_times_from_external_list(settings)
    return get_start_times_overlapping_windows(
        settings["start_time"],
        settings["end_time"],
        settings["window_length"],
        settings["overlap"],
    )


def get_start_times_overlapping_windows(start_time, end_time, window_length, overlap):
    """ Generate start times for given parameters

    :param start_time: start time of first time window
    :type start_time: str or obspy.UTCDateTime()
    :param end_time: end time of last time window
    :type end_time: str or obspy.UTCDateTime()
    :param window_length: length of time window in seconds
    :type window_length: float
    :param overlap: percentage of time window overlap
    :type overlap: float
    :return: list of start times
    :rtype: list[obspy.UTCDateTime()]
    """
    from numpy import linspace
    from obspy import UTCDateTime

    if not 0 <= overlap < 1:
        raise ValueError("overlap must be 0 <= overlap < 1")

    if not isinstance(start_time, UTCDateTime):
        start_time = UTCDateTime(start_time)
    if not isinstance(end_time, UTCDateTime):
        end_time = UTCDateTime(end_time)

    total_time_frame = end_time - start_time
    n_windows = (total_time_frame / window_length) / (1 - overlap)
    window_times = linspace(0, total_time_frame, int(n_windows))
    # remove time windows that are too short at the end
    window_times = [wt for wt in window_times if wt + window_length < total_time_frame]
    # "convert" to UTCDateTime
    start_times = [start_time + wt for wt in window_times]
    return start_times


def get_start_times_from_external_list(settings):
    """
    Extract start times for MFP analysis from external file,
    specified in settings["external_timelist"].

    :param settings: dict holding all info for project
    :type settings: dict
    :return: list of start times specified in external file
    :rtype: list[UTCDateTime]
    """
    import pandas as pd
    from obspy import UTCDateTime

    df = pd.read_csv(settings["external_timelist"])
    return [UTCDateTime(_) for _ in df["time"]]

