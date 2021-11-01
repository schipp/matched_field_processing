import numpy as np
from tqdm import tqdm

from .misc import chunks


def bartlett_processor(csdm, gf_spectra):
    """Computes the beampower using the Bartlett Processor.
    Other names: Delay-and-Sum Beamformer, Conventional Beamformer, Frequency Beamformer
    

    :param csdm: [description]
    :type csdm: [type]
    :param gf_spectra: [description]
    :type gf_spectra: [type]
    :return: [description]
    :rtype: [type]
    """

    return np.real(np.einsum("ik,ijk,jk", np.conj(gf_spectra), csdm, gf_spectra))


def L2_processor(data_traces, gf_traces):
    # time
    # print(np.diagonal(csdm, axis1=0, axis2=1).shape)
    # csdm_auto = np.sqrt(np.diagonal(csdm, axis1=0, axis2=1).T)
    individual_bps = []
    for dtrace, gtrace in zip(data_traces, gf_traces):
        L2norm = np.linalg.norm(np.abs(dtrace - gtrace), ord=2)
        individual_bps.append(L2norm)
    beampower = np.median(individual_bps)
    return beampower


def wasserstein_processor(data_traces, gf_traces):
    from scipy.stats import wasserstein_distance

    for dtrace, gtrace in zip(data_traces, gf_traces):
        dist = wasserstein_distance(dtrace, gtrace)
        individual_bps.append(dist)
    beampower = np.median(individual_bps)


def get_beampowers(csdm, gf_spectra, gp_dists, gp_on_land, relevant_dists, settings):
    """
    Computes the Beampower for all grid-points.
    """

    # how to parallelise - multiprocessing?
    # create object that each process writes into

    def get_bp(
        d,
        chunk_idxs,
        dists_azs_chunk,
        csdm,
        gf_spectra,
        gp_dists,
        relevant_dists,
        gp_on_land,
        settings,
        n_proc,
    ):

        # for chunk_idx, dists_azs in tqdm(
        #     zip(chunk_idxs, dists_azs_chunk),
        #     desc=f"beampower process: {n_proc}",
        #     total=len(chunk_idxs),
        #     position=n_proc,
        #     mininterval=2,
        # ):
        for chunk_idx, dists_azs in zip(chunk_idxs, dists_azs_chunk):

            # check if gp is on land and skip
            if settings["exclude_land"] and gp_on_land[chunk_idx]:
                d[chunk_idx] = np.nan
                continue

            dist_idxs = np.array(
                [
                    np.where(
                        (relevant_dists[:, 0] == dist_az[0])
                        & (relevant_dists[:, 1] == dist_az[1])
                    )[0]
                    for dist_az in dists_azs
                ]
            ).flatten()
            # print(dist_idxs)
            # print(dist_idxs.shape)
            wut_idx = np.array(
                [
                    np.array([_ + i for i, __ in enumerate(settings["components"])])
                    for _ in dist_idxs
                ]
            ).flatten()
            gf_spectra_relevant = gf_spectra[wut_idx]
            # gf_traces_relevant = gf_traces[wut_idx]

            # # weigh by distance (surface wave propagation)
            # for _idx, d_a in enumerate(dists_azs):
            #     gf_spectra_relevant[_idx, :] /= d_a[0]**2

            # limit to distance range
            if settings["use_max_dist"]:
                for _idx, d_a in enumerate(dists_azs):
                    if d_a[0] >= settings["max_dist"] or d_a[0] <= settings["min_dist"]:
                        gf_spectra_relevant[_idx, :] = np.zeros(
                            gf_spectra_relevant.shape[1]
                        ).astype(complex)

            # print(f'{csdm.shape=}')
            # print(f'{gf_spectra_relevant.shape=}')

            if settings["do_iteration"]:
                tmp_beampowers = []

                if settings["iteration_type"] == "monte_carlo":
                    for pre_factor in np.random.uniform(-1e3, 1e3, size=100):
                        # gf_spectra_scaled = pre_factor*gf_spectra_relevant.real + gf_spectra_relevant.imag*1j

                        beampower = bartlett_processor(
                            csdm=csdm, gf_spectra=gf_spectra_scaled
                        )
                        # beampower = L2_processor(data_traces, pre_factor*gf_traces_relevant)
                        tmp_beampowers.append(beampower)
                elif settings["iteration_type"] == "gradient_descent":
                    pass

                d[chunk_idx] = np.max(tmp_beampowers, axis=0)
            else:
                # d[chunk_idx] = MVDR(csdm=csdm, gf_spectra=gf_spectra_relevant)
                d[chunk_idx] = bartlett_processor(
                    csdm=csdm, gf_spectra=gf_spectra_relevant
                )
                # d[chunk_idx] = L2_processor(data_traces, gf_traces_relevant)

    from multiprocessing import Manager, Process

    # n_processes = 20
    # print(len(gp_dists))
    chunk_gen = chunks(gp_dists, settings["n_processes"])
    # print(list(chunk_gen)[0])
    import sys

    # sys.exit()
    with Manager() as manager:
        d = manager.dict()

        processes = []
        for n_proc, (chunk_idxs, dists_azs_chunk) in enumerate(chunk_gen):
            p = Process(
                target=get_bp,
                args=(
                    d,
                    chunk_idxs,
                    dists_azs_chunk,
                    csdm,
                    gf_spectra,
                    gp_dists,
                    relevant_dists,
                    gp_on_land,
                    settings,
                    n_proc,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # reshape shared dictionary into list of beampowers
        beampowers = []
        for d_idx in range(len(gp_dists)):
            beampowers.append(d[d_idx])

    # beampowers = []
    # for dists_azs in tqdm(gp_dists, desc='beampowers'):
    #     # dists = get_distances(list_of_locs=station_locations, point=gp)
    #     # dists = np.round(dists, decimals=decimal_round)

    #     # dist_idxs = np.array([np.where(np.isin(relevant_dists, dist)) for dist in dists]).flatten()
    #     # gf_spectra_relevant = gf_spectra[dist_idxs]

    #     dist_idxs = np.array([np.where((relevant_dists[:,0] == dist_az[0]) & (relevant_dists[:,1] == dist_az[1]))[0] for dist_az in dists_azs]).flatten()
    #     # print(dist_idxs)
    #     # print(dist_idxs.shape)
    #     wut_idx = np.array([np.array([_+i for i, __ in enumerate(settings['components'])]) for _ in dist_idxs]).flatten()
    #     gf_spectra_relevant = gf_spectra[wut_idx]

    #     # print(f'{csdm.shape=}')
    #     # print(f'{gf_spectra_relevant.shape=}')

    #     beampower = bartlett_processor(csdm=csdm, gf_spectra=gf_spectra_relevant)

    #     beampowers.append(beampower)

    return beampowers


def get_beampowers_in_cut_timewindows(
    st, gf_spectra, gp_dists, relevant_dists, start_time, idx_for_label
):
    """
    Computes the Beampower for all grid-points.
    """

    # compute spectra for all data
    from scipy.fftpack import fft

    beampowers = []
    for dists in tqdm(
        gp_dists, desc=f"beampowers {idx_for_label[0]+1}/{idx_for_label[1]}"
    ):

        data_spectra = []
        # replace trim by time window based on arrival
        for tr_idx, tr in enumerate(st):
            distance = dists[tr_idx]
            # trim tr to given time window
            ss = start_time - tr.stats.starttime
            idx_start = int(
                tr.stats.sampling_rate
                * (ss + distance / vel - window_length_around_rayleigh_arrival)
            )
            idx_end = int(
                tr.stats.sampling_rate
                * (ss + distance / vel + window_length_around_rayleigh_arrival)
            )
            data = tr.data[idx_start:idx_end]

            tr_spectrum = fft(data)
            spec_freqs = fftfreq(len(data), sampling_rate)
            spec_freqs_of_interest_idx = (spec_freqs >= fmin) & (spec_freqs <= fmax)
            data_spectra.append(tr_spectrum[spec_freqs_of_interest_idx])

        data_spectra = np.array(data_spectra)
        csdm = get_csdm(spectra=data_spectra)

        # dists = get_distances(list_of_locs=station_locations, point=gp)
        # dists = np.round(dists, decimals=decimal_round)

        # dist_idxs = np.array([np.where(np.isin(relevant_dists, dist)) for dist in dists]).flatten()
        # gf_spectra_relevant = gf_spectra[dist_idxs]

        beampower = bartlett_processor(csdm=csdm, gf_spectra=data_spectra)

        beampowers.append(beampower)

    return beampowers


def get_beampowers_w_synths(
    csdm, gf_spectra, gp_dists, relevant_dists, start_time, settings
):
    """
    Computes the Beampower for all grid-points.
    """

    # compute spectra for all data
    # from scipy.fftpack import fft

    beampowers = []
    # grid_points
    for dists_azs in tqdm(gp_dists, desc="beampowers"):

        # compute synthetic spectra

        # dists = get_distances(list_of_locs=station_locations, point=gp)
        # dists = np.round(dists, decimals=decimal_round)
        # dist_idxs = np.array([np.where(np.isin(relevant_dists, dist_az)) for dist_az in dists_azs]).flatten()
        # print(relevant_dists)
        # dist_idxs = []
        # for dist_az in dists_azs:
        #     __ = np.where((relevant_dists[:,0] == dist_az[0]) & (relevant_dists[:,1] == dist_az[1]))[0]
        #     if len(__) == 0:
        #         print('__')
        #         print('__')
        #         print(dist_az)
        #         print('__')
        #     dist_idxs.append(__)
        dist_idxs = np.array(
            [
                np.where(
                    (relevant_dists[:, 0] == dist_az[0])
                    & (relevant_dists[:, 1] == dist_az[1])
                )[0]
                for dist_az in dists_azs
            ]
        ).flatten()
        # print(dist_idxs)
        # print(dist_idxs.shape)
        wut_idx = np.array(
            [
                np.array([_ + i for i, __ in enumerate(settings["components"])])
                for _ in dist_idxs
            ]
        ).flatten()
        gf_spectra_relevant = gf_spectra[wut_idx]

        beampower = bartlett_processor(csdm=csdm, gf_spectra=gf_spectra_relevant)

        beampowers.append(beampower)

    return beampowers
