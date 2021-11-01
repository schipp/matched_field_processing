# Matched Field Processing using full Green's Functions from pre-computed databases

Precise knowledge of the sources of seismic noise is fundamental to our understanding of the ambient seismic field. Two approaches to locating sources exist currently. One is based on comparing estimated Green's functions from cross-correlation of seismic noise with synthetically computed correlation functions. This approach is computationally expensive and not yet widely adopted. The other, more common approach is Beamforming, where a beam is computed by shifting waveforms in time corresponding to a potential slowness of an arriving wave front. Beamforming allows fast computations, but samples the slowness domain, thus limiting it to the plane-wave assumption and sources outside of the array.

Matched Field Processing (MFP) is Beamforming in the spatial domain. By probing potential source locations directly, it allows for arbitrary wave propagation in the medium and sources inside of arrays. MFP has been successfully applied on local scale using constant velocity models, and regional scale using travel times estimated from phase velocity maps. At global scale, a constant velocity model is inadequate and phase velocity maps have not yet been published for periods below ~20s.

To advance MFP towards the global scale, we replace the need for travel-time information with full synthetic Green's functions. This allows to capture the full complexity of wave propagation by including amplitude information and multiple phases. 

This repository is the development platform for the method described above.

## TODO

- [ ] Clean-up for publication.
    - [ ] Commit current status
    - [ ] Remove unused code sections
    - [ ] Remove unecessary functions and corresponding logic
        - [ ] SVD?
        - [ ] Coherency-weighting?
    - [ ] Add support for depth from config
    - [ ] Improve amplitude correction logic
    - [ ] Check all comments
- [ ] Rewrite to use acoular.
### Requirements

- python 3.8+
- obspy
- cartopy
- tqdm
- pyyaml
- instaseis
- local green's function database, e.g., via syngine
