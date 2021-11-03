import numpy as np
from tqdm import tqdm

from .misc import chunks


def conventional_beamformer(csdm: np.ndarray, gf_spectra: np.ndarray) -> float:
    """
    Computes the beampower using the conventional_beamformer.
    Other names: Delay-and-Sum Beamformer, Bartlett Processor, Frequency Beamformer

    :param csdm: [description]
    :type csdm: [type]
    :param gf_spectra: [description]
    :type gf_spectra: [type]
    :return: [description]
    :rtype: [type]
    """

    return np.real(np.einsum("ik,ijk,jk", np.conj(gf_spectra), csdm, gf_spectra))


def get_beampowers(
    csdm: np.ndarray,
    gf_spectra: np.ndarray,
    gp_dists: np.ndarray,
    gp_on_land: np.ndarray,
    relevant_dists: np.ndarray,
    settings: dict,
) -> np.ndarray:
    """Compute the Beampower for all grid-points, distributed over multiple processes for speed.

    :param csdm: [description]
    :type csdm: np.ndarray
    :param gf_spectra: [description]
    :type gf_spectra: np.ndarray
    :param gp_dists: [description]
    :type gp_dists: np.ndarray
    :param gp_on_land: [description]
    :type gp_on_land: np.ndarray
    :param relevant_dists: [description]
    :type relevant_dists: np.ndarray
    :param settings: [description]
    :type settings: dict
    :return: [description]
    :rtype: np.ndarray
    """

    def get_bp(
        d: dict,
        cell_idxs: np.ndarray,
        cell_dists_azs: np.ndarray,
        csdm: np.ndarray,
        gf_spectra: np.ndarray,
        relevant_dists: np.ndarray,
        gp_on_land: np.ndarray,
        settings: dict,
        n_proc: int,
    ) -> None:
        """ Compute beampower for all cells assigned to the process.

        :param d: shared-memory dictionary to hold beampowers for all cells. 
        :type d: dict
        :param cell_idxs: indices of the assigned cells.
        :type cell_idxs: np.ndarray
        :param cell_dists_azs: distances and azimuths of assigned cells.
        :type cell_dists_azs: np.ndarray
        :param csdm: cross-spectral density matrix
        :type csdm: np.ndarray
        :param gf_spectra: all green's functions
        :type gf_spectra: np.ndarray
        :param relevant_dists: all relevant (=unique) distances
        :type relevant_dists: np.ndarray
        :param gp_on_land: info which cells are located on-land. can be skipped via settings["exclude_land"]
        :type gp_on_land: np.ndarray
        :param settings: dict holding all global paramters for current run
        :type settings: dict
        :param n_proc: which process this is assigned to
        :type n_proc: int
        """

        for cell_idx, dists_azs in tqdm(
            zip(cell_idxs, cell_dists_azs),
            desc=f"beampower process: {n_proc}",
            total=len(cell_idxs),
            position=n_proc + 1,
            mininterval=2,
            leave=False,
            # disable=~settings["verbose"],
        ):
            # check if gp is on land and skip computation
            if (
                settings["geometry_type"] == "geographic"
                and settings["exclude_land"]
                and gp_on_land[cell_idx]
            ):
                d[cell_idx] = np.nan
                continue

            # select the right green's functions for the current stations' distances and azimuths
            dist_idxs = np.array(
                [
                    np.where(
                        (relevant_dists[:, 0] == dist_az[0])
                        & (relevant_dists[:, 1] == dist_az[1])
                    )[0]
                    for dist_az in dists_azs
                ]
            ).flatten()

            # handle multiple components
            # INFO: will result in bugs atm.
            # DO NOT USE MULTIPLE COMPONENTS ATM
            wut_idx = np.array(
                [
                    np.array([_ + i for i, __ in enumerate(settings["components"])])
                    for _ in dist_idxs
                ]
            ).flatten()

            # extract correct green's functions for current distances/azimuths
            gf_spectra_relevant = gf_spectra[wut_idx]

            d[cell_idx] = conventional_beamformer(
                csdm=csdm, gf_spectra=gf_spectra_relevant
            )

    # multiprocessing logic starts here
    from multiprocessing import Manager, Process

    chunk_gen = chunks(gp_dists, settings["n_processes"])

    with Manager() as manager:
        d = manager.dict()

        processes = []
        for n_proc, (cell_idxs, cell_dists_azs) in enumerate(chunk_gen):
            p = Process(
                target=get_bp,
                args=(
                    d,
                    cell_idxs,
                    cell_dists_azs,
                    csdm,
                    gf_spectra,
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

    return beampowers
