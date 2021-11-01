import numpy as np

def get_start_times(settings):
    if settings['use_external_timelist']:
        return get_start_times_from_external_list(settings)
    return get_start_times_overlapping_windows(settings['start_time'], settings['end_time'], settings['window_length'], settings['overlap'])

def get_start_times_overlapping_windows(start_time, end_time, window_length, overlap):
    from obspy import UTCDateTime
    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)

    total_time_frame = end_time - start_time
    n_windows = (total_time_frame / window_length) / (1 - overlap)
    window_times = np.linspace(0, total_time_frame, int(n_windows))
    # remove time windows that are too short at the end
    window_times = [wt for wt in window_times if wt + window_length < total_time_frame]
    # "convert" to UTCDateTime
    start_times = [start_time + wt for wt in window_times]
    return start_times

def get_start_times_from_external_list(settings):
    import pandas as pd
    from obspy import UTCDateTime
    df = pd.read_csv(settings['external_timelist'])
    return [UTCDateTime(_) for _ in df['time']]