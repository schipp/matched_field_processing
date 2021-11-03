import logging


def get_data_spectra(st, start_time, settings, fp, freqs_of_interest_idx):
    import numpy as np
    from scipy.fftpack import fft, fftfreq

    # real data here
    st_curr = st.slice(
        starttime=start_time, endtime=start_time + settings["window_length"],
    )

    logging.info(st_curr)

    data_spectra = []
    for tr in st_curr:
        data = tr.data.copy()
        # ensure that data is right length
        if len(data) < settings["window_length"]:
            logging.info(
                f"Data too short ({len(data)})for {tr.stats.station}. Filling with {settings['window_length'] - len(data)} 0s"
            )
            # fill with needed zeros
            data = np.append(
                data, np.array([0] * (settings["window_length"] - len(data)))
            )

        if len(data) > settings["window_length"]:
            logging.debug(
                f"Data too long ({len(data)})for {tr.stats.station}. Removing {len(data) - settings['window_length']}last samples"
            )
            data = data[: settings["window_length"]]

        tr_spectrum = fft(data)

        spec_freqs = fftfreq(len(data), 1 / settings["sampling_rate"])
        if freqs_of_interest_idx is None:
            data_spectrum = tr_spectrum
        else:
            spec_freqs_of_interest_idx = (spec_freqs >= fp[0]) & (spec_freqs <= fp[1])
            data_spectrum = tr_spectrum[spec_freqs_of_interest_idx]

        data_spectra.append(data_spectrum)

    return np.array(data_spectra)
