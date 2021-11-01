import numpy as np


def get_csdm(spectra: np.ndarray) -> np.ndarray:
    """
    Compute the cross-spectral density-matrix (CSDM)
    spectra: (N x M) shaped np.ndarray with N recorded spectra of length M
    """

    # einstein convention version -- memory intensive
    csdm = np.einsum("ik,jk->ijk", spectra, np.conj(spectra))

    # print(spectra.shape)

    # make csdm strictly upper-triangular to remove redudancy
    zero_spectra = np.zeros(csdm.shape[2]).astype(complex)
    for idx in range(csdm.shape[0]):
        for jdx in range(csdm.shape[1]):
            if idx == jdx:
                csdm[idx, jdx, :] = zero_spectra
    # csdm = np.triu(csdm, k=1)

    # # old slow version
    # csdm = np.zeros((n_stations, n_stations, len(freqs[freqs_of_interest_idx])), dtype=np.complex)
    # for i, spec1 in enumerate(source_spectra):
    #     for j, spec2 in enumerate(source_spectra):
    #         csdm[i, j] = spec1 * np.conj(spec2)

    return csdm


def svd_csdm(csdm, n_components, settings):
    csdm_shapes = csdm.shape
    u, s, vh = np.linalg.svd(
        csdm.reshape(csdm_shapes[2], csdm_shapes[0], csdm_shapes[1])
    )

    csdm = []
    for u_i, s_i, vh_i in zip(u, s, vh):
        # split into "high" and "low" ??
        # csdm_i = u_i * s_i * vh_i
        reduced_csdm = (
            np.matrix(u_i[:, n_components:])
            * np.diag(s_i[n_components:])
            * np.matrix(vh_i[n_components:, :])
        )
        csdm.append(reduced_csdm)
        # spectra_red = (np.identity(3, 3) - u_i * vh_i) * spectra
    # np.einsum('ik,jk->ijk', spectra_red, np.conj(spectra_red))
    csdm = np.array(csdm).reshape(csdm_shapes)
    return csdm


def weigh_spectra_by_coherency(spectra):
    from itertools import permutations

    corr_coefs = np.zeros((len(spectra), len(spectra)))
    for i, j in permutations(range(len(spectra)), 2):
        # ccoeff = np.corrcoef(spectra[i], spectra[j])
        # print(ccoeff)
        corr_coefs[i, j] = np.corrcoef(spectra[i], spectra[j])[0, 0]

        # L2 norm
        # corr_coefs[i, j] = np.linalg.norm(spectra[i] - spectra[j], ord=2)

    mean_corrcoefs = np.mean(corr_coefs, axis=0) + np.mean(corr_coefs, axis=1)

    spectra /= mean_corrcoefs[:, np.newaxis] + 2

    # for idx, spectrum in enumerate(spectra):
    #     # number of zeros are the same for every station, can be ignored?
    #     # mean_corrcoef = np.mean(corr_coefs[idx, :] + corr_coefs[:, idx])

    #     # shift coeff +2, so that min is 1 (keep if anticorrelated to all)
    #     # spectrum /= mean_corrcoef + 2

    #     # shift coeff +2, so that min is 1 (keep if anticorrelated to all)
    #     spectrum /= mean_corrcoef

    return spectra
    # for spectrum in spectra:
    #     np.correlate(spectra)
