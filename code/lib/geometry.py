import numpy as np
from tqdm import tqdm
from .misc import chunks
from .synth import get_synth_stations, get_real_station_locations_worldwide

import logging

def generate_global_grid(settings):
    """
    Generate the grid geometry
    Returns coordinates of n grid_points as [[x0,y0,z0], [x1,y1,z1], ..., [xn,yn,zn]|
    """
    from itertools import product

    if settings['geometry_type'] == 'cartesian':
        grid_limits_lon = settings['grid_limits_x']
        grid_limits_lat = settings['grid_limits_y']
    else:
        grid_limits_lon = settings['grid_limits_lon']
        grid_limits_lat = settings['grid_limits_lat']

    grid_spacing_in_deg = settings['grid_spacing']

    n_gridpoints_lon = int((grid_limits_lon[1] - grid_limits_lon[0]) / grid_spacing_in_deg)
    n_gridpoints_lat = int((grid_limits_lat[1] - grid_limits_lat[0]) / grid_spacing_in_deg)
    
    # grid geometry
    grid_lon_coords = np.linspace(grid_limits_lon[0], grid_limits_lon[1], n_gridpoints_lon)
    grid_lat_coords = np.linspace(grid_limits_lat[0], grid_limits_lat[1], n_gridpoints_lat)
    grid_points = np.asarray(list(product(grid_lon_coords, grid_lat_coords)))

    # exclude grid points on land?
    # from global_land_mask import globe
    # for gp in grid_points:
    #     if globe.is_land(gp[1], gp[0])

    return grid_points, grid_lon_coords, grid_lat_coords


def get_distances(settings, list_of_locs:np.ndarray, point:np.ndarray) -> np.ndarray:
    """
    Compute the distance between an array of coordinate-pairs and a single point.
    If you want 3D distances (source at depth) list_of_locs = [(lat1, lon1, alt1), ...]
    """
    # cartesian
    if settings['geometry_type'] == 'cartesian':
        dists = np.linalg.norm(list_of_locs - point, ord=2, axis=1)
        azs = np.zeros(len(dists))
        return np.array(list(zip(dists, azs)))

    # geometric
    from obspy.geodetics import gps2dist_azimuth

    dists_azs = [gps2dist_azimuth(
        lat1=sta_loc[1],
        lon1=sta_loc[0],
        lat2=point[1],
        lon2=point[0]
        )[:2] for sta_loc in list_of_locs]

    dists_azs = np.array(dists_azs)
    dists_azs[:, 0] /= 1000  # to km
    
    return np.array(dists_azs)


def get_all_distances_rounded(station_locations, grid_points, settings) -> list:
    """
    Computes the distances between all station locations and grid_points.
    """

    save_f = f"{settings['project_dir']}/out/gp_dists.npy"
    
    import os
    if os.path.isfile(save_f):
        sl_load, gps_load, gp_dists = np.load(save_f, allow_pickle=True)
        logging.info(f'station_locs: {sl_load.shape} vs. {station_locations.shape}')
        logging.info(f'grid_points: {gps_load.shape} vs. {grid_points.shape}')
        if gps_load.shape == grid_points.shape and \
            sl_load.shape == station_locations.shape:
            if (sl_load == station_locations).all() and \
                (gps_load == grid_points).all():
                return gp_dists

    from multiprocessing import Process, Manager
    # n_processes = 20
    # print(len(gp_dists))
    chunk_gen = chunks(grid_points, settings['n_processes'])

    def get_dist_az(gp_dists_azs_d, gp_chunk, chunk_idxs, station_locations, n_proc, settings):
        for chunk_idx, gp in tqdm(zip(chunk_idxs, gp_chunk), desc=f'grid-station distances {n_proc}', total=len(chunk_idxs), position=n_proc, mininterval=2):
            dists_azs = get_distances(settings=settings, list_of_locs=station_locations, point=gp)
            # round to deciamsl to reduce number of required synth spectra
            dists = np.round(dists_azs[:, 0], decimals=settings['decimal_round'])
            azs = np.round(dists_azs[:, 1], decimals=0)

            daz = np.array(list(zip(dists, azs)))

            gp_dists_azs_d[chunk_idx] = daz

    with Manager() as manager:
        # gp_dists = []
        gp_dists_azs_d = manager.dict()

        processes = []
        for n_proc, (chunk_idxs, gp_chunk) in enumerate(chunk_gen):
            p = Process(
                target=get_dist_az,
                args=(gp_dists_azs_d, gp_chunk, chunk_idxs, station_locations, n_proc, settings)
                )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()

        # reshape shared dictionary into list of beampowers
        gp_dists_azs = []
        for d_idx in range(len(grid_points)):
            gp_dists_azs.append(gp_dists_azs_d[d_idx])

    gp_dists_azs = np.array(gp_dists_azs)
    # get_distances
    np.save(save_f, [station_locations, grid_points, gp_dists_azs])
    logging.info(f'Successfully saved {save_f}')
    
    return gp_dists_azs

def get_station_locs_from_staxml(st, settings):
    import obspy
    station_locations = []
    for tr in st:
        inv = obspy.read_inventory(f"{settings['sta_xml_dir']}/{tr.stats.network}.{tr.stats.station}.xml", format='STATIONXML')
        lat, lon = inv[0][0][0].latitude, inv[0][0][0].longitude
        station_locations.append([lon, lat])
    return np.array(station_locations)

def get_station_locations(settings, st):
    # get station locations
    if settings['use_synth_stations']:
        return get_synth_stations(settings, wiggle=0)
    else:
        if settings['use_open_worldwide_stations']:
            return get_real_station_locations_worldwide()
        return get_station_locs_from_staxml(st, settings=settings)

def get_loc_from_timelist(time, settings):
    import pandas as pd
    df = pd.read_csv(settings['external_timelist'])

    lons, lats = df['lon'].values, df['lat'].values
    from obspy import UTCDateTime
    times = np.array([UTCDateTime(t) for t in df['time'].values])

    return lons[np.argmin(np.abs(times-time))], lats[np.argmin(np.abs(times-time))]