import obspy
import os
inv = obspy.Inventory()
for fn in os.listdir('../data/stations'):
    if '.xml' in fn:
        inv += obspy.read_inventory(f'../data/stations/{fn}', format='STATIONXML')



# read downloaded data
# wf_dir = '../data/waveforms_global_chile_unsanitized/'
# wf_dir = '../data/waveforms_CI_NoCal/'
# wf_dir = '../data/waveforms_arrays/'
# wf_dir = '../data/waveforms_atlantic_2020_01_01/'
# wf_dir = '../data/waveforms_atlantic_2020_01_01_min10km/'
# wf_dir = '../data/waveforms_ridge_eq_2019_02_14_arrays/'
# wf_dir = '../data/waveforms_north_sea_eq_all/'
# wf_dir = '../data/waveforms_atlantic_larger_region/'
# wf_dir = '../data/waveforms_atlantic_eq3/'
# wf_dir = '../data/waveforms_northsea_2014_02/'
wf_dir = '../data/waveforms_northsea_2019_02/'


st = obspy.read(f'{wf_dir}/*.mseed')
# st = obspy.read(f'{wf_dir}/NO.*HZ*.mseed')
# st = obspy.read(f'{wf_dir}/GR.GR*HZ*.mseed')
# st = obspy.read(f'{wf_dir}/WI.*HZ*.mseed')
# st = obspy.read(f'{wf_dir}/CN.*HZ*.mseed')
# st = obspy.read(f'{wf_dir}/IV.*HZ*.mseed')
# st = obspy.read(f'{wf_dir}/TA.*HZ*.mseed')

# outdir = '../data/deconv/northsea_2014_02/'
outdir = '../data/deconv/northsea_2019_02/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# REMOVE RESPONSE
pre_filt = (0.005, 0.006, 250.0, 400.0)

import numpy as np
stations = np.unique([_.split('.')[1] for _ in os.listdir(f'{wf_dir}')])

from tqdm import tqdm

# st_all = obspy.Stream()
# import pyasdf
# ds = pyasdf.ASDFDataSet(f"{outdir}/all.h5", compression="gzip-3")
for station in tqdm(stations):
    try:
        
        # if inv.select(station=station)[0][0][0].latitude < 15:
        #    # print('-–', station)
        #     continue

        trs = st.select(station=station)
        trs.merge(fill_value=0)
        # trs.merge()
        # if len(trs) != 3:
        #     print(f'Skipping {station} - {len(trs)} channels')
        #     continue
        trs.resample(1)

        trs.remove_response(inventory=inv, output='DISP', pre_filt=pre_filt)
        trs.detrend('demean')
        trs.detrend('linear')

        # for tr in trs:
        #     st_dec = obspy.Stream()
        #     if tr.stats.sampling_rate != 1:
        #         print(f'Resampling {tr.stats.station}')
        #         tr.resample(1)
            
        #     if tr.stats.channel == 'LH1':
        #         tr.stats.channel = 'LHN'
        #     elif tr.stats.channel == 'LH2':
        #         tr.stats.channel = 'LHE'

        #     st_dec += tr
        #     st_dec.write(f'../data/deconv/{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}.MSEED')
        # st_all += trs
        trs.write(f'{outdir}/{trs[0].stats.network}.{trs[0].stats.station}.mseed')
    except ValueError as m:
        print(m)
        print(f'Skipping {station} - no response file')
    except:
        print(f'Skipping {station} - unknown error')