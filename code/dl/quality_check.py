import obspy
import numpy as np

# st = obspy.read('../data/deconv/northsea_2014_02/*.mseed')
st = obspy.read('../data/deconv/northsea_2019_02/*.mseed')
print(st)

# amps = np.array([np.max(np.abs(tr.data)) for tr in st])
# zero_ratio = [np.count_nonzero(tr.data)/len(tr.data) for tr in st]
# stds = [np.std(tr.data) for tr in st]
# energies = [np.linalg.norm(tr.data, ord=2) for tr in st]
# outliers = np.digitize(amps, bins=(np.linspace(0, 1e-3, 100))) > 75
# fig.savefig('outlier_hist.png')

# print(np.percentile(amps, 2))
# print(np.percentile(amps, 98))

amps = np.array([np.max(np.abs(tr.data)) for tr in st])
amp_std = np.std(amps)
stds = np.array([np.std(tr.data) for tr in st])
nonzero_ratios = [np.count_nonzero(tr.data)/len(tr.data) for tr in st]
energies = np.array([np.linalg.norm(tr.data, ord=2) for tr in st])

# print(amp_std)
# print(amps)
# print(amps - amp_std)
amps_good = (amps<0.05) & (amps>0.0001)
std_good = (stds>1E-6) & (stds<1E-4)
print(np.where((amps_good) & (std_good))[0])
print(len(amps))
print(len(amps[(amps_good) & (std_good)]))

st_new = obspy.Stream()
for idx, tr in enumerate(st):
    if not idx in np.where((amps_good) & (std_good))[0]:
        print(tr)
        continue
    st_new += tr

st_slice = st_new.slice(starttime=obspy.UTCDateTime('2019-02-10T00:00:00.0Z'), endtime=obspy.UTCDateTime('2019-02-10T05:00:00.0Z'))
import pylab as plt
fig, axs = plt.subplots(len(st_slice), 1, figsize=(4, 30))
fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1)
max_amp = np.max([np.max(np.abs(tr.data)) for tr in st_new])
for tr, ax in zip(st_slice, axs):
    ax.plot(tr.times(), tr.data)
    ax.set_frame_on(False)
    ax.set_ylim(-max_amp, max_amp)
    ax.set_xticks([])
fig.savefig('outlier_traces.png', dpi=300)
plt.close(fig)
# print(st[np.where((amps_good) & (std_good))[0]])
#print(np.where(amps>10))

fig, axs = plt.subplots(1, 3)
axs[2].plot(stds, 'o')
axs[2].plot(stds[(amps_good) & (std_good)], 'o')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[1].plot(energies, 'o')
axs[1].plot(energies[(amps_good) & (std_good)], 'o')
axs[1].set_xscale('log')
axs[1].set_yscale('log')

ax = axs[0]
ax.plot(amps, 'o')
ax.plot(amps[(amps_good) & (std_good)], 'o')
ax.set_xscale('log')
ax.set_yscale('log')
# ax.axhline(10)
# ax.axhline(0.0001)
# ax.axhline(amps)
fig.savefig('outlier_method.png')

# print(np.percentile(zero_ratio, 2))
# print(np.percentile(zero_ratio, 98))
# print(zero_ratio)
# print(stds)
# print(energies)
# print(np.percentile(energies, 2))
# print(np.percentile(energies, 98))
# print(amps > 2e2)
# print(amps < 8e1)
# inliers = 1e2 < amps < 2e2
# print(inliers)

# st_new = obspy.Stream()
# amps = np.array([np.max(np.abs(tr.data)) for tr in st])
# zero_ratio = [np.count_nonzero(tr.data)/len(tr.data) for tr in st]
# stds = [np.std(tr.data) for tr in st]
# energies = [np.linalg.norm(tr.data, ord=2) for tr in st]
# from tqdm import tqdm
# for tr, amp, energy, std in tqdm(zip(st, amps, energies, stds), total=len(st)):
#     if amp <= np.percentile(amps, 2) or amp >= np.percentile(amps, 98):
#         print('skipping ', tr.stats.station, ' amp')
#         continue
#     if energy <= np.percentile(energies, 2) or energy >= np.percentile(energies, 98):
#         print('skipping ', tr.stats.station, ' energy')
#         continue
#     if std <= np.percentile(stds, 2) or std >= np.percentile(stds, 98):
#         print('skipping ', tr.stats.station, ' std')
#         continue
#     tr.write(f'../data/deconv/northsea_2014_02/outlier/{tr.stats.network}.{tr.stats.station}.mseed')