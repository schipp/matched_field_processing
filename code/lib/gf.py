import numpy as np
from scipy.fftpack import ifft
from tqdm import tqdm

from .misc import chunks


def get_tt_from_spectrum(spectrum, settings):
    """
    Measure the traveltime of the maximum of the envelope from a given specturm.
    Thus, returns group-traveltime.
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


def get_steering_spectrum(freqs, traveltime):
    return np.exp(-1j * 2 * np.pi * freqs * traveltime)


def get_gf_spectrum(
    dist_az: float,
    settings,
    use_instaseis: bool = True,
    sampling_rate: float = 1.0,
    instaseis_db=None,
    is_synth=False,
    sdr=False,
) -> np.ndarray:
    """
    Compute the Green's Function (GF) spectrum for given frequencies, distance, and medium-velocity.
    Replace by more complex GFs if 1D, 2D or 3D model is available.
    """

    # using precomputed GF (instaseis)

    from obspy.geodetics import kilometers2degrees

    st = instaseis_db.get_greens_function(
        epicentral_distance_in_degree=kilometers2degrees(dist_az[0]),
        source_depth_in_m=0,
        dt=sampling_rate,
    )

    # relevant GFs for vertical
    ZSS = st.select(channel="ZSS")[0].data
    ZDS = st.select(channel="ZDS")[0].data
    ZDD = st.select(channel="ZDD")[0].data
    ZEP = st.select(channel="ZEP")[0].data
    RSS = st.select(channel="RSS")[0].data
    RDS = st.select(channel="RDS")[0].data
    RDD = st.select(channel="RDD")[0].data
    REP = st.select(channel="REP")[0].data
    TSS = st.select(channel="TSS")[0].data
    TDS = st.select(channel="TDS")[0].data

    # explosion mt
    if settings["strike_dip_rake_gridsearch"] and sdr:
        from obspy.imaging.scripts.mopad import MomentTensor

        mt = np.asarray(MomentTensor(sdr).get_M())
        Mxx, Mxy, Mxz = mt[0]
        Myy, Myz = mt[1, 1:]
        Mzz = mt[2, -1]
        # Mxx, Myy, Mzz, Mxy, Mxz, Myz
    else:
        Mxx, Myy, Mzz, Mxy, Mxz, Myz = settings["MT"]

    # Mxx = 1
    # Myy = 1
    # Mzz = 1
    # Mxy = 0
    # Mxz = 0
    # Myz = 0

    # angle does not matter for 'explosion' source
    # as Mxx and Myy contribution negate each other, if Mxx=Myy
    #
    # we don't compute GFs for all necessary angle-dist combinations,
    # but instead rotate seismograms before filling up gf_spectra
    a = np.radians(dist_az[1])

    # compute vertical trace from individual green's functions
    uz = (
        Mxx * (np.cos(2 * a) * ZSS / 2 - ZDD / 6 + ZEP / 3)
        + Myy * (-np.cos(2 * a) * ZSS / 2 - ZDD / 6 + ZEP / 3)
        + Mzz * (ZDD / 3 + ZEP / 3)
        + Mxy * (ZSS * np.sin(2 * a))
        + Mxz * (ZDS * np.cos(a))
        + Myz * (ZDS * np.sin(a))
    )

    # compute radial trace from individual green's functions
    ur = (
        Mxx * (np.cos(2 * a) * RSS / 2 - RDD / 6 + REP / 3)
        + Myy * (-np.cos(2 * a) * RSS / 2 - RDD / 6 + REP / 3)
        + Mzz * (RDD / 3 + REP / 3)
        + Mxy * (RSS * np.sin(2 * a))
        + Mxz * (RDS * np.cos(a))
        + Myz * (RDS * np.sin(a))
    )

    # compute transversal trace from individual green's functions
    ut = (
        Mxx * (np.sin(2 * a) * TSS / 2)
        + Myy * (-np.sin(2 * a) * TSS / 2)
        + Mxy * (-TSS * np.cos(2 * a))
        + Mxz * (TDS * np.sin(a))
        + Myz * (-TDS * np.cos(a))
    )

    from scipy.fftpack import fft, fftfreq, ifft

    # freqs_ = fftfreq(n=len(st[0].data), d=sampling_rate)
    # normalize
    if not is_synth:
        if settings["norm_mode_gf"] == "time_domain":
            if settings["normalize_gf"]:
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

    spec_z = fft(uz)
    spec_r = fft(ur)
    spec_t = fft(ut)

    def whiten_spectrum(spec):
        # compute energy
        E = np.sqrt(np.sum(np.absolute(np.square(spec)))) / (len(spec))
        # take only angles (phase) and multiply by normalized energy
        spec = np.exp(1j * np.angle(spec)) * E
        return spec

    if not is_synth:
        if settings["norm_mode_gf"] == "whitening":
            if settings["normalize_gf"]:
                spec_z = whiten_spectrum(spec_z)
                spec_r = whiten_spectrum(spec_r)
                spec_t = whiten_spectrum(spec_t)

                spec_z /= np.max(np.abs(spec_z))
                spec_r /= np.max(np.abs(spec_r))
                spec_t /= np.max(np.abs(spec_t))
        if settings["amplitude_treatment"] == "spreading_attenuation":
            dist = dist_az[0]
            v = 2600
            Q = 57822.0  # PREM Qkappa
            Q = 1e6
            Q = 1e12
            Q = 1e20
            Q = 60000
            Q = 6
            # Q = 100  # PREM Qkappa
            # Q = 600.0 # PREM Qmu
            omega = 2 * np.pi * np.fft.fftfreq(len(uz), 1 / sampling_rate)
            # np.fft.fftf# omega = uz
            # Bowden et al. 2020 - Appendix A
            A = np.sqrt((2 * v) / (np.pi * omega * dist)) * np.exp(
                (-omega * dist) / (2 * v * Q)
                + np.sqrt((2) / (np.pi * dist)) * np.exp(-1j * np.pi / 4)
            )
            # Corciulo et al 2012 - Equation 3
            # A = np.sqrt((2) / (np.pi * dist)) * np.exp(-1j * np.pi / 4)
            spec_z /= A
            spec_r /= A
            spec_t /= A

    from scipy.signal import hilbert

    if settings["do_energy"]:
        spec_z = fft(np.abs(hilbert(np.abs(ifft(spec_z)))) ** 2)
        spec_r = fft(np.abs(hilbert(np.abs(ifft(spec_r)))) ** 2)
        spec_t = fft(np.abs(hilbert(np.abs(ifft(spec_t)))) ** 2)

    return spec_z, spec_r, spec_t

    # only phase
    # return np.exp(-1j * freqs * dist/vel)
    # surface wave decay 1/r^2 -> heavy artifacts/impact near stations
    # return np.sqrt(2 / (np.pi*dist)) * np.exp(-1j * freqs * dist/vel)


def get_gf_traces(
    dist_az: float,
    settings,
    use_instaseis: bool = True,
    sampling_rate: float = 1.0,
    instaseis_db=None,
    sdr=False,
) -> np.ndarray:
    """
    Compute the Green's Function (GF) spectrum for given frequencies, distance, and medium-velocity.
    Replace by more complex GFs if 1D, 2D or 3D model is available.
    """

    # using precomputed GF (instaseis)

    from obspy.geodetics import kilometers2degrees

    st = instaseis_db.get_greens_function(
        epicentral_distance_in_degree=kilometers2degrees(dist_az[0]),
        source_depth_in_m=0,
        dt=sampling_rate,
    )

    # relevant GFs for vertical
    ZSS = st.select(channel="ZSS")[0].data
    ZDS = st.select(channel="ZDS")[0].data
    ZDD = st.select(channel="ZDD")[0].data
    ZEP = st.select(channel="ZEP")[0].data
    RSS = st.select(channel="RSS")[0].data
    RDS = st.select(channel="RDS")[0].data
    RDD = st.select(channel="RDD")[0].data
    REP = st.select(channel="REP")[0].data
    TSS = st.select(channel="TSS")[0].data
    TDS = st.select(channel="TDS")[0].data

    # explosion mt
    if settings["strike_dip_rake_gridsearch"]:
        from obspy.imaging.scripts.mopad import MomentTensor

        mt = np.asarray(MomentTensor(sdr).get_M())
        Mxx, Mxy, Mxz = mt[0]
        Myy, Myz = mt[1, 1:]
        Mzz = mt[2, -1]
    else:
        Mxx, Myy, Mzz, Mxy, Mxz, Myz = settings["MT"]

    # Mxx = 1
    # Myy = 1
    # Mzz = 1
    # Mxy = 0
    # Mxz = 0
    # Myz = 0

    # angle does not matter for 'explosion' source
    # as Mxx and Myy contribution negate each other, if Mxx=Myy
    #
    # we don't compute GFs for all necessary angle-dist combinations,
    # but instead rotate seismograms before filling up gf_spectra
    a = np.radians(dist_az[1])

    # compute vertical trace from individual green's functions
    uz = (
        Mxx * (np.cos(2 * a) * ZSS / 2 - ZDD / 6 + ZEP / 3)
        + Myy * (-np.cos(2 * a) * ZSS / 2 - ZDD / 6 + ZEP / 3)
        + Mzz * (ZDD / 3 + ZEP / 3)
        + Mxy * (ZSS * np.sin(2 * a))
        + Mxz * (ZDS * np.cos(a))
        + Myz * (ZDS * np.sin(a))
    )

    # compute radial trace from individual green's functions
    ur = (
        Mxx * (np.cos(2 * a) * RSS / 2 - RDD / 6 + REP / 3)
        + Myy * (-np.cos(2 * a) * RSS / 2 - RDD / 6 + REP / 3)
        + Mzz * (RDD / 3 + REP / 3)
        + Mxy * (RSS * np.sin(2 * a))
        + Mxz * (RDS * np.cos(a))
        + Myz * (RDS * np.sin(a))
    )

    # compute transversal trace from individual green's functions
    ut = (
        Mxx * (np.sin(2 * a) * TSS / 2)
        + Myy * (-np.sin(2 * a) * TSS / 2)
        + Mxy * (-TSS * np.cos(2 * a))
        + Mxz * (TDS * np.sin(a))
        + Myz * (-TDS * np.cos(a))
    )

    from scipy.fftpack import fft, fftfreq

    # freqs_ = fftfreq(n=len(st[0].data), d=sampling_rate)
    # normalize
    if settings["normalize_gf"]:
        uz /= np.max(np.abs(uz))
        ur /= np.max(np.abs(ur))
        ut /= np.max(np.abs(ut))

    return uz, ur, ut

    # only phase
    # return np.exp(-1j * freqs * dist/vel)
    # surface wave decay 1/r^2 -> heavy artifacts/impact near stations
    # return np.sqrt(2 / (np.pi*dist)) * np.exp(-1j * freqs * dist/vel)


def get_gf_spectra_for_dists(
    freqs, dists_azs, instaseis_db, settings, sdr=False, save_out=True
) -> np.ndarray:
    """
    Computes the Green's Functions spectra for relevant distances only
    """
    import pickle

    save_f = f"{settings['project_dir']}/out/gfs.pkl"

    import os

    if os.path.isfile(save_f):
        # gf_spectra, freqs_load, dists_load = np.load(save_f, allow_pickle=True)
        with open(save_f, "rb") as sf:
            gf_spectra, freqs_load, dists_load = pickle.load(sf)

        if freqs_load.shape == freqs.shape and dists_load.shape == dists_azs.shape:
            if (freqs_load == freqs).all() and (dists_load == dists_azs).all():
                return gf_spectra

    from multiprocessing import Manager, Process

    chunk_gen = chunks(dists_azs, settings["n_processes"])

    # gf_spectra = []
    def get_gf(gf_spectra_d, dists_azs_chunk, chunk_idxs, instaseis_db, n_proc, sdr):
        # for chunk_idx, dist_az in tqdm(
        #     zip(chunk_idxs, dists_azs_chunk),
        #     desc="computing necessary GFs",
        #     position=n_proc,
        #     total=len(dists_azs_chunk),
        # ):
        for chunk_idx, dist_az in zip(chunk_idxs, dists_azs_chunk):
            spec_z, spec_r, spec_t = get_gf_spectrum(
                dist_az, settings, instaseis_db=instaseis_db, sdr=sdr
            )
            if settings["type_of_gf"] == "GF":
                # if 'Z' in settings['components']:
                #     gf_spectra.append(spec_z)
                # if 'R' in settings['components']:
                #     gf_spectra.append(spec_r)
                # if 'T' in settings['components']:
                #     gf_spectra.append(spec_t)
                gf_spectra_d[chunk_idx] = spec_z
            elif settings["type_of_gf"] == "measure_from_GF":
                traveltime = get_tt_from_spectrum(spec_z, settings)
                gf_spectra_d[chunk_idx] = get_steering_spectrum(freqs, traveltime)
            elif settings["type_of_gf"] == "v_const":
                traveltime = dist_az[0] / settings["v_const"]
                gf_spectra_d[chunk_idx] = get_steering_spectrum(freqs, traveltime)

    with Manager() as manager:
        # gp_dists = []
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

    # gp_dists_azs = np.array(gp_dists_azs)
    gf_spectra = np.array(gf_spectra)

    if save_out and not settings["strike_dip_rake_gridsearch"]:
        with open(save_f, "wb") as sf:
            pickle.dump([gf_spectra, freqs, dists_azs], sf)
    # np.save(save_f, )

    return gf_spectra


def get_gf_traces_for_dists(freqs, dists_azs, instaseis_db, settings) -> np.ndarray:
    """
    Computes the Green's Functions spectra for relevant distances only
    """
    import pickle

    save_f = f"{settings['project_dir']}/out/gfs_traces.pkl"

    import os

    if os.path.isfile(save_f):
        # gf_spectra, freqs_load, dists_load = np.load(save_f, allow_pickle=True)
        with open(save_f, "rb") as sf:
            gf_spectra, freqs_load, dists_load = pickle.load(sf)

        if freqs_load.shape == freqs.shape and dists_load.shape == dists_azs.shape:
            if (freqs_load == freqs).all() and (dists_load == dists_azs).all():
                return gf_spectra

    from multiprocessing import Manager, Process

    chunk_gen = chunks(dists_azs, settings["n_processes"])

    # gf_spectra = []
    def get_gf(gf_spectra_d, dists_azs_chunk, chunk_idxs, instaseis_db, n_proc):
        # for chunk_idx, dist_az in tqdm(
        #     zip(chunk_idxs, dists_azs_chunk),
        #     desc="computing necessary GFs",
        #     position=n_proc,
        #     total=len(dists_azs_chunk),
        # ):
        for chunk_idx, dist_az in zip(chunk_idxs, dists_azs_chunk):
            uz, ur, ut = get_gf_traces(dist_az, settings, instaseis_db=instaseis_db)
            if settings["type_of_gf"] == "GF":
                # if 'Z' in settings['components']:
                #     gf_spectra.append(spec_z)
                # if 'R' in settings['components']:
                #     gf_spectra.append(spec_r)
                # if 'T' in settings['components']:
                #     gf_spectra.append(spec_t)
                gf_spectra_d[chunk_idx] = uz
            elif settings["type_of_gf"] == "measure_from_GF":
                traveltime = get_tt_from_spectrum(spec_z, settings)
                gf_spectra_d[chunk_idx] = get_steering_spectrum(freqs, traveltime)
            elif settings["type_of_gf"] == "v_const":
                traveltime = dist_az[0] / settings["v_const"]
                gf_spectra_d[chunk_idx] = get_steering_spectrum(freqs, traveltime)

    with Manager() as manager:
        # gp_dists = []
        gf_spectra_d = manager.dict()

        processes = []
        for n_proc, (chunk_idxs, dists_azs_chunk) in enumerate(chunk_gen):
            p = Process(
                target=get_gf,
                args=(gf_spectra_d, dists_azs_chunk, chunk_idxs, instaseis_db, n_proc),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # reshape shared dictionary into list of beampowers
        gf_spectra = []
        for d_idx in range(len(dists_azs)):
            gf_spectra.append(gf_spectra_d[d_idx])

    # gp_dists_azs = np.array(gp_dists_azs)
    gf_spectra = np.array(gf_spectra)

    with open(save_f, "wb") as sf:
        pickle.dump([gf_spectra, freqs, dists_azs], sf)
    # np.save(save_f, )

    return gf_spectra
