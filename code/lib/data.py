import logging


def get_data_spectra(st, start_time, settings, fp, freqs_of_interest_idx):
    import numpy as np

    # real data here
    st_curr = st.copy()
    st_curr.trim(starttime=start_time, endtime=start_time + settings["window_length"])

    logging.info(st_curr)

    # compute spectra for all data
    from scipy.fftpack import fft, fftfreq

    data_spectra = []

    from scipy.signal import hilbert

    # replace trim by time window based on arrival
    for tr in st_curr:
        # trim tr to given time window
        # tr_start = tr.stats.starttime
        # idx_start = tr.stats.samplingrate * (tr_start + distance / vel - window_length_around_rayleigh_arrival)
        # idx_end = tr.stats.samplingrate * (tr_start + distance / vel + window_length_around_rayleigh_arrival)
        # data = tr.data[idx_start:idx_end]

        # data = tr.data[:-1]
        data = tr.data
        if len(data) < settings["window_length"]:
            logging.warning(
                f"Data too short ({len(data)})for {tr.stats.station}. Filling with {settings['window_length'] - len(data)} 0s"
            )
            # fill with needed zeros
            data = np.append(
                data, np.array([0] * (settings["window_length"] - len(data)))
            )
        if len(data) > settings["window_length"]:
            logging.warning(
                f"Data too long ({len(data)})for {tr.stats.station}. Removing {len(data) - settings['window_length']}last samples"
            )
            data = data[: settings["window_length"] - 1]
        if settings["do_energy"]:
            data = np.abs(hilbert(data)) ** 2

        tr_spectrum = fft(data)
        spec_freqs = fftfreq(len(data), settings["sampling_rate"])
        # normalize for each station individually
        # data_spectrum = tr_spectrum[spec_freqs_of_interest_idx]
        if freqs_of_interest_idx is None:
            data_spectrum = tr_spectrum
        else:
            spec_freqs_of_interest_idx = (spec_freqs >= fp[0]) & (spec_freqs <= fp[1])
            data_spectrum = tr_spectrum[spec_freqs_of_interest_idx]
        # /np.max(np.abs(tr_spectrum[spec_freqs_of_interest_idx]))
        data_spectra.append(data_spectrum)

    return np.array(data_spectra)


def get_data_traces(st, start_time, fp, settings):
    import numpy as np

    # real data here
    st_curr = st.copy()
    st_curr.filter("bandpass", freqmin=fp[0], freqmax=fp[1])
    n_stations_before_trim = len(st_curr)
    st_curr.trim(starttime=start_time, endtime=start_time + settings["window_length"])
    n_stations_after_trim = len(st_curr)
    if n_stations_after_trim < n_stations_before_trim:
        stations_in_st_curr = [tr.stats.station for tr in st_curr]
        removed_traces = [
            tr.stats.station for tr in st if tr.stats.station not in stations_in_st_curr
        ]
        logging.warning(f"Data removed from window: {removed_traces}")
    logging.info(st_curr)

    if settings["do_synth"]:
        traces = np.array([tr.data[:-1] for tr in st_curr])
    else:
        traces = np.array([tr.data for tr in st_curr])

    return traces
