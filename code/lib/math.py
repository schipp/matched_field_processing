import numpy as np


def get_csdm(spectra: np.ndarray) -> np.ndarray:
    """Compute the cross-spectral density-matrix (CSDM)
    spectra: 

    :param spectra: spectra of all stations. (N x ω) shape with N recorded spectra of length ω.
    :type spectra: np.ndarray
    :return: cross-spectral density matrix. shape (N x N x ω)
    :rtype: np.ndarray
    """

    # Einstein convention version -- memory intensive, but fast.
    csdm = np.einsum("ik,jk->ijk", spectra, np.conj(spectra))

    # exclude auto correlations.
    zero_spectra = np.zeros(csdm.shape[2]).astype(complex)
    for idx in range(csdm.shape[0]):
        csdm[idx, idx, :] = zero_spectra

    # # old slow version -- may be used for JIT compilation optimization(?)
    # csdm = np.zeros((n_stations, n_stations, len(freqs[freqs_of_interest_idx])), dtype=np.complex)
    # for i, spec1 in enumerate(source_spectra):
    #     for j, spec2 in enumerate(source_spectra):
    #         csdm[i, j] = spec1 * np.conj(spec2)

    return csdm


def svd_csdm(csdm: np.ndarray, n_components: int) -> np.ndarray:
    """ Initial implementation of singular-value-decomposition and 
    reduction of cross-spectral density matrix.
    
    This is used for subspace beamformers and may help detect multiple sources.
    The current formulation here differs from Corciulo, M., Roux, P., Campillo, M., Dubucq, D. &#38; Kuperman, W.A. (2012) Multiscale matched-field processing for noise-source localization in exploration geophysics. <i>GEOPHYSICS</i>, <b>77</b>, KS33–KS41. doi:10.1190/geo2011-0438.1

    !! IMPORTANT: WIP, does not appear to be correct at the moment !!

    :param csdm: cross-spectral density matrix
    :type csdm: np.ndarray
    :param n_components: how many eigenvector to keep
    :type n_components: int
    :return: reduced cross-spectral density matrix, where only the first n eigenvectors are kept
    :rtype: np.ndarray
    """

    csdm_shape = csdm.shape
    u, s, vh = np.linalg.svd(csdm.reshape(csdm_shape[2], csdm_shape[0], csdm_shape[1]))

    csdm = []
    for u_i, s_i, vh_i in zip(u, s, vh):
        # split into "high" and "low"
        reduced_csdm = (
            np.matrix(u_i[:, n_components:])
            * np.diag(s_i[n_components:])
            * np.matrix(vh_i[n_components:, :])
        )
        csdm.append(reduced_csdm)
    csdm = np.array(csdm).reshape(csdm_shape)
    return csdm
