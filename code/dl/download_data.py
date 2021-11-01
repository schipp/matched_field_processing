# import yaml
# with open('metadata.yml') as f:
#     metadata = yaml.load(f, Loader=yaml.FullLoader)

# download data near trajectory (use visible end location for now)
import obspy
from obspy.clients.fdsn import Client

clients = []
for provider in ['BGR','EMSC','ETH','GEONET','GFZ','ICGC','INGV','IPGP','IRIS','ISC','KNMI','KOERI','LMU','NCEDC','NIEP','NOA','ODC','ORFEUS','RASPISHAKE','RESIF','SCEDC','TEXNET','USGS','USP', 'http://eida.geo.uib.no']:
# for provider in ['BGR']:
    clients.append(Client(provider))
starttime = obspy.UTCDateTime('2019-09-29T15:00:00.0Z')
endtime = obspy.UTCDateTime('2019-09-29T23:00:00.0Z')

# northern cali eq
starttime = obspy.UTCDateTime('2014-08-24T00:00:00Z')
endtime = obspy.UTCDateTime('2014-08-24T23:59:59Z')

# atlantic ridge eq 1
starttime = obspy.UTCDateTime('2019-01-01T00:00:00Z')
endtime = obspy.UTCDateTime('2019-01-30T23:59:59Z')

# atlantic ridge eq 2
# with USARRAY - MW5.4 Northern Mid-Atlantic Ridge - 45.0251° N 	27.9662° W - 2014-05-23T23:41:48Z
# starttime = obspy.UTCDateTime('2014-05-23T00:00:01Z')
# endtime = obspy.UTCDateTime('2014-05-24T23:59:59Z')

# NORTH SEA event 1 - 56.97N, 1.83E, 2km
# 2019-09-24T13:38:12.7
# starttime = obspy.UTCDateTime('2019-09-24T00:00:00Z')
# endtime = obspy.UTCDateTime('2019-09-24T23:59:59Z')

# WINTER 2014 CONTINUOUS
starttime = obspy.UTCDateTime('2014-02-01T00:00:01Z')
endtime = obspy.UTCDateTime('2014-02-28T23:59:59Z')

# WINTER 2019 CONTINUOUS
starttime = obspy.UTCDateTime('2019-02-01T00:00:01Z')
endtime = obspy.UTCDateTime('2019-02-28T23:59:59Z')



# chino hills - 29 July 2008
# starttime = obspy.UTCDateTime('2008-07-29T00:00:00.0Z')
# endtime = obspy.UTCDateTime('2008-07-30T00:00:00.0Z')
# https://www.emsc-csem.org/Earthquake/earthquake.php?id=794224#map
from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader, RectangularDomain

# client.set_eida_token('eidatoken.txt')


# grid_limits_lon: [-65, 14]
# grid_limits_lat: [25, 65]
# domain = GlobalDomain()
# lon0, lon1 = -75, 5
# lat0, lat1 = 15, 75

# Atlantic larger
# lon0, lon1 = -110, 14
# lat0, lat1 = -5, 75

# roughly north sea region
# lon0, lon1 = -14, 17
# lat0, lat1 = 42, 65
# roughly north sea region incl. iceland
lon0, lon1 = -22, 18
lat0, lat1 = 35, 66
domain = RectangularDomain(minlatitude=lat0, maxlatitude=lat1, minlongitude=lon0, maxlongitude=lon1)

# inv = obspy.read_inventory('../data/phoenix_TA.xml')
# print(inv)
# with open('array_stationlist.csv') as f:
#     for line in f.readlines():
#         network = line.split(',')[0]
#         stations = line.split(',')[1:-1]
#         sta_str = ','.join(stations)

#         print(network, sta_str)

restrictions = Restrictions(
    starttime=starttime,
    endtime=endtime,
    # limit_stations_to_inventory=inv,
    # network='*',
    # network='CI',
    network='*',
    station='*',
    # network='CN',
    # station='SVNB,LMN,ELNB,HKNB,SRNB,WCNB,BOIN,HSNB,GGN',
    # station='S*',
    # network='UU',
    # network='G,GE',
    reject_channels_with_gaps=False,
    minimum_length=0.99,
    # minimum_interstation_distance_in_m=25E3,
    # channel="LH*",
    channel_priorities=["LHZ", "BHZ"],  # , "BHZ", "HHZ"]  # , "EHZ", "SHZ"],
    sanitize=False,
    # exclude_networks=['Z3']
    )

# No specified providers will result in all known ones being queried.
mdl = MassDownloader(clients)
# The data will be downloaded to the ``./waveforms/`` and ``./stations/``
wf_dir = '../data/waveforms_northsea_2019_02/'

import os
if not os.path.exists(wf_dir):
    os.mkdir(wf_dir)

mdl.download(domain, restrictions, mseed_storage=wf_dir, stationxml_storage="../data/stations")
