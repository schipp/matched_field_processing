from logging import disable

import instaseis
import numpy as np

from .misc import chunks


def get_tt_from_spectrum(spectrum: np.ndarray, settings: dict) -> float:
    """
    Measure the traveltime of the maximum of the envelope from a given specturm.
    Thus, returns group-traveltime, if surface wave (depends on distance).

    :param spectrum: spectrum to measure travel-time from
    :type spectrum: np.ndarray
    :param settings: dict containing all global parameters for current run.
    :type settings: dict
    :return: [description]
    :rtype: float
    """
    from scipy.fftpack import ifft

    trace = ifft(spectrum).real
    from obspy.signal.filter import bandpass
    from scipy.signal import hilbert

    trace_filt = bandpass(
        trace,
        freqmin=settings["fmin"],
        freqmax=settings["fmax"],
        df=settings["sampling_rate"],
    )
    analytic_signal = hilbert(trace_filt)
    amplitude_envelope = np.abs(analytic_signal)
    times = np.arange(
        0,
        len(trace_filt) / settings["sampling_rate"] + 1 / settings["sampling_rate"],
        1 / settings["sampling_rate"],
    )
    return times[np.argmax(amplitude_envelope)]


def get_steering_spectrum(freqs: np.ndarray, traveltime: float) -> np.ndarray:
    """ Compute the steering spectrum for the given travel time.
    This is the standard approach to MFP, and can have many names:
    Steering vector, replica vectore, synthetic wave field, Green's function.

    :param freqs: frequencies to compute the spectrum for
    :type freqs: np.ndarray
    :param traveltime: travel time (or phase shift)
    :type traveltime: float
    :return: Steering vector
    :rtype: np.ndarray
    """
    return np.exp(-1j * 2 * np.pi * freqs * traveltime)


def get_gf_spectrum(
    dist_az: list,
    settings: dict,
    instaseis_db,
    sampling_rate: float = 1.0,
    is_synth: bool = False,
    sdr: bool or list = False,
) -> np.ndarray:
    """
    Compute the Green's Function (GF) spectra for given distance and azimuth dist_az.
    This works by extracting the base Green's functions from the specified instaseis DB,
    imposing a moment tensor, computing the seismograms, 
    applying any processing specified in settings and, converting them to spectra.

    :param dist_az: distance and azimuth to compute the GF for.
    :type dist_az: list or np.ndarray
    :param settings: dict containing all global parameters for current run.
    :type settings: dict
    :param instaseis_db: open connection to an instaseis database that GFs are extracted from.
    :type instaseis_db: instaseis.instaseisDB
    :param sampling_rate: sampling rate of data used, defaults to 1.0
    :type sampling_rate: float, optional
    :param is_synth: Toggle to check whether GF computed is for synthetic data, defaults to False
    :type is_synth: bool, optional
    :param sdr: strike-dip-rake to compute tensor for, defaults to False. Used only when strike-dip-rake is being grid-searched.
    :type sdr: bool or list, optional
    :return: Green's function spectrum
    :rtype: np.ndarray
    """

    from obspy.geodetics import kilometers2degrees

    # relevant base GFs
    st = instaseis_db.get_greens_function(
        epicentral_distance_in_degree=kilometers2degrees(dist_az[0]),
        source_depth_in_m=settings["source_depth"],
        dt=sampling_rate,
    )

    # for vertical component seismogram
    ZSS = st.select(channel="ZSS")[0].data
    ZDS = st.select(channel="ZDS")[0].data
    ZDD = st.select(channel="ZDD")[0].data
    ZEP = st.select(channel="ZEP")[0].data
    # for radial component seismogram
    RSS = st.select(channel="RSS")[0].data
    RDS = st.select(channel="RDS")[0].data
    RDD = st.select(channel="RDD")[0].data
    REP = st.select(channel="REP")[0].data
    # for transverse component seismogram
    TSS = st.select(channel="TSS")[0].data
    TDS = st.select(channel="TDS")[0].data

    if settings["strike_dip_rake_gridsearch"] and sdr:
        from obspy.imaging.scripts.mopad import MomentTensor

        mt = np.asarray(MomentTensor(sdr).get_M())
        Mxx, Mxy, Mxz = mt[0]
        Myy, Myz = mt[1, 1:]
        Mzz = mt[2, -1]
    else:
        Mxx, Myy, Mzz, Mxy, Mxz, Myz = settings["MT"]

    # angle does not matter for 'explosion' source
    # as Mxx and Myy contribution negate each other, if Mxx=Myy
    a = np.radians(dist_az[1])

    # vertical seismogram from base GFs
    uz = (
        Mxx * (np.cos(2 * a) * ZSS / 2 - ZDD / 6 + ZEP / 3)
        + Myy * (-np.cos(2 * a) * ZSS / 2 - ZDD / 6 + ZEP / 3)
        + Mzz * (ZDD / 3 + ZEP / 3)
        + Mxy * (ZSS * np.sin(2 * a))
        + Mxz * (ZDS * np.cos(a))
        + Myz * (ZDS * np.sin(a))
    )

    # radial seismogram from base GFs
    ur = (
        Mxx * (np.cos(2 * a) * RSS / 2 - RDD / 6 + REP / 3)
        + Myy * (-np.cos(2 * a) * RSS / 2 - RDD / 6 + REP / 3)
        + Mzz * (RDD / 3 + REP / 3)
        + Mxy * (RSS * np.sin(2 * a))
        + Mxz * (RDS * np.cos(a))
        + Myz * (RDS * np.sin(a))
    )

    # transversal seismogram from base GFs
    ut = (
        Mxx * (np.sin(2 * a) * TSS / 2)
        + Myy * (-np.sin(2 * a) * TSS / 2)
        + Mxy * (-TSS * np.cos(2 * a))
        + Mxz * (TDS * np.sin(a))
        + Myz * (-TDS * np.cos(a))
    )

    from scipy.fftpack import fft, ifft

    # processing of seismograms (in time-domain)
    if not is_synth:
        if settings["amplitude_treatment"] == "time_domain_norm":
            uz /= np.max(np.abs(uz))
            ur /= np.max(np.abs(ur))
            ut /= np.max(np.abs(ut))

    if is_synth and settings["add_noise_to_synth"] > 0:
        noise_bounds_z = settings["add_noise_to_synth"] * np.max(np.abs(uz))
        noise_bounds_r = settings["add_noise_to_synth"] * np.max(np.abs(ur))
        noise_bounds_t = settings["add_noise_to_synth"] * np.max(np.abs(ut))
        uz += np.random.uniform(low=-noise_bounds_z, high=noise_bounds_z, size=len(uz))
        ur += np.random.uniform(low=-noise_bounds_r, high=noise_bounds_r, size=len(ur))
        ut += np.random.uniform(low=-noise_bounds_t, high=noise_bounds_t, size=len(ut))

    # processing of spectra (in frequency-domain)
    spec_z = fft(uz)
    spec_r = fft(ur)
    spec_t = fft(ut)

    def whiten_spectrum(spec):
        """ Performs spectral whitening on given spectrum.
        Whitening is weighting all frequencies equally.
        This is the strictest possible version. No smoothing or waterlevel.

        :param spec: Spectrum to be whitened
        :type spec: np.ndarray
        :return: Whitened spectrum
        :rtype: np.ndarray
        """
        import numpy as np

        # compute energy density of spectrum
        # E = np.sqrt(np.sum(np.absolute(np.square(spec)))) / (len(spec))
        # strict whitening is discarding spectral amplitudes
        spec = np.exp(1j * np.angle(spec))
        return spec

    if not is_synth:
        if settings["amplitude_treatment"] == "whitening_GF":
            spec_z = whiten_spectrum(spec_z)
            spec_r = whiten_spectrum(spec_r)
            spec_t = whiten_spectrum(spec_t)
        if settings["amplitude_treatment"] == "spreading_attenuation":
            dist = dist_az[0]
            v = 2600
            Q = 6
            omega = 2 * np.pi * np.fft.fftfreq(len(uz), 1 / sampling_rate)
            # Corciulo et al 2012 - Equation 3 + Bowden et al. 2020 - Appendix A
            A = np.sqrt((2 * v) / (np.pi * omega * dist)) * np.exp(
                (-omega * dist) / (2 * v * Q)
            ) + np.sqrt((2) / (np.pi * dist)) * np.exp(-1j * np.pi / 4)
            spec_z /= A
            spec_r /= A
            spec_t /= A

    return spec_z, spec_r, spec_t


def get_gf_spectra_for_dists(
    freqs: np.ndarray,
    dists_azs: list,
    instaseis_db,
    settings: dict,
    sdr=False,
    save_out=True,
) -> np.ndarray:
    """
    Computes the Green's Functions spectra for relevant distances only using multiprocessing.

    :param freqs: [description]
    :type freqs: np.ndarray
    :param dists_azs: [description]
    :type dists_azs: list
    :param instaseis_db: [description]
    :type instaseis_db: instaseis.InstaseisDB
    :param settings: [description]
    :type settings: dict
    :param sdr: [description], defaults to False
    :type sdr: bool, optional
    :param save_out: [description], defaults to True
    :type save_out: bool, optional
    :return: [description]
    :rtype: np.ndarray
    """
    import os
    import pickle
    from multiprocessing import Manager, Process

    from tqdm import tqdm

    save_f = f"{settings['project_dir']}/out/gfs.pkl"
    # has this run been done before?
    if os.path.isfile(save_f):
        with open(save_f, "rb") as sf:
            gf_spectra, freqs_load, dists_load = pickle.load(sf)

        # some sanity checks whether the previously saved data could be correct
        if freqs_load.shape == freqs.shape and dists_load.shape == dists_azs.shape:
            if (freqs_load == freqs).all() and (dists_load == dists_azs).all():
                return gf_spectra

    chunk_gen = chunks(dists_azs, settings["n_processes"])

    # gf_spectra = []
    def get_gf(gf_spectra_d, dists_azs_chunk, chunk_idxs, instaseis_db, n_proc, sdr):
        for chunk_idx, dist_az in tqdm(
            zip(chunk_idxs, dists_azs_chunk),
            desc="computing necessary GFs",
            position=n_proc + 1,
            total=len(dists_azs_chunk),
            leave=False,
            # disable=~settings["verbose"],
        ):
            # currently, radial and transverse components are computed but neglected.
            # this has only minor impact on performance, as its done only once.
            spec_z, spec_r, spec_t = get_gf_spectrum(
                dist_az, settings, instaseis_db=instaseis_db, sdr=sdr
            )
            if settings["type_of_gf"] == "GF":
                gf_spectra_d[chunk_idx] = spec_z
            elif settings["type_of_gf"] == "measure_from_GF":
                traveltime = get_tt_from_spectrum(spec_z, settings)
                gf_spectra_d[chunk_idx] = get_steering_spectrum(freqs, traveltime)
            elif settings["type_of_gf"] == "v_const":
                traveltime = dist_az[0] / settings["v_const"]
                gf_spectra_d[chunk_idx] = get_steering_spectrum(freqs, traveltime)

    # multiprocessing logic here
    # using a shared dictionary between processes to save individual green's functions
    with Manager() as manager:
        gf_spectra_d = manager.dict()

        processes = []
        for n_proc, (chunk_idxs, dists_azs_chunk) in enumerate(chunk_gen):
            p = Process(
                target=get_gf,
                args=(
                    gf_spectra_d,
                    dists_azs_chunk,
                    chunk_idxs,
                    instaseis_db,
                    n_proc,
                    sdr,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # reshape shared dictionary into list of beampowers
        gf_spectra = []
        for d_idx in range(len(dists_azs)):
            gf_spectra.append(gf_spectra_d[d_idx])

    gf_spectra = np.array(gf_spectra)

    # output only saved if not grid searching, this would generate too much (unecessary) data.
    if save_out and not settings["strike_dip_rake_gridsearch"]:
        with open(save_f, "wb") as sf:
            pickle.dump([gf_spectra, freqs, dists_azs], sf)

    return gf_spectra
