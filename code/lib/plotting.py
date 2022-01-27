import logging

import numpy as np

from .geometry import get_loc_from_timelist


def plot_results(
    beampowers: np.ndarray,
    settings: dict,
    start_time: "obspy.UTCDateTime",
    grid_lon_coords: np.ndarray,
    grid_lat_coords: np.ndarray,
    station_locations: np.ndarray,
    plot_identifier: str = "empty",
) -> None:

    # normalise windows for plotting
    beampowers_to_plot = beampowers / np.nanmax(np.abs(beampowers))

    # extract source location
    if settings["do_synth"]:
        source_loc = np.array(settings["synth_sources"]).T
    elif not settings["do_synth"] and settings["time_window_mode"] == "file":
        source_loc = get_loc_from_timelist(start_time, settings)
    else:
        source_loc = np.nan, np.nan
    pcm = plot_beampowers_on_map(
        lons=grid_lon_coords,
        lats=grid_lat_coords,
        beampowers=beampowers_to_plot,
        station_locations=station_locations,
        settings=settings,
        outfile=f"{settings['project_basedir']}/{settings['project_id']}/plots/{start_time.timestamp}_{plot_identifier}.png",
        source_loc=source_loc,
        title=start_time,
        plot_station_locations=True,
    )


def plot_beampowers_on_map(
    lons,
    lats,
    beampowers,
    settings,
    station_locations=None,
    outfile=None,
    source_loc=None,
    title=None,
    fig=None,
    ax=None,
    vmin=None,
    vmax=None,
    plot_station_locations=False,
):
    """ Plots beampower distribution.

    :param lons: [description]
    :type lons: [type]
    :param lats: [description]
    :type lats: [type]
    :param beampowers: [description]
    :type beampowers: [type]
    :param settings: [description]
    :type settings: [type]
    :param station_locations: [description], defaults to None
    :type station_locations: [type], optional
    :param outfile: [description], defaults to None
    :type outfile: [type], optional
    :param source_loc: [description], defaults to None
    :type source_loc: [type], optional
    :param title: [description], defaults to None
    :type title: [type], optional
    :param fig: [description], defaults to None
    :type fig: [type], optional
    :param ax: [description], defaults to None
    :type ax: [type], optional
    :param vmin: [description], defaults to None
    :type vmin: [type], optional
    :param vmax: [description], defaults to None
    :type vmax: [type], optional
    :param plot_station_locations: [description], defaults to False
    :type plot_station_locations: bool, optional
    :return: [description]
    :rtype: [type]
    """

    import cartopy.crs as ccrs
    import pylab as plt
    from cmcrameri import cm

    plt.style.use("dracula")

    # crs_use = ccrs.Robinson()
    # convert lons, lats to corresponding CRS

    xx, yy = np.meshgrid(lons, lats)

    # ax = plt.axes(projection=ccrs.Orthographic(central_longitude=source_loc[0], central_latitude=source_loc[1]))

    if fig == None and ax == None:
        if settings["geometry_type"] == "geographic":
            fig = plt.figure(figsize=(8, 4))
            ax = plt.axes(projection=ccrs.Robinson())
            fig.subplots_adjust(left=0)
        elif settings["geometry_type"] == "cartesian":
            fig = plt.figure(figsize=(5, 4))
            ax = plt.axes()
            ax.set_aspect("equal")

    # _map = Basemap(projection='eck4',lon_0=0,resolution='c', ax=ax)
    # _map.drawcoastlines(linewidth=.5, color='k')
    # _map.drawparallels(np.arange(-90.,120.,30.))
    # _map.drawmeridians(np.arange(0.,420.,60.))z
    trans = ccrs.PlateCarree()

    if settings["geometry_type"] == "geographic":
        ax.coastlines(resolution="10m", linewidth=0.5, color="#AAAAAA")

    if plot_station_locations:
        if settings["geometry_type"] == "geographic":
            ax.scatter(
                station_locations[:, 0],
                station_locations[:, 1],
                c="k",
                s=4,
                marker="^",
                lw=0,
                transform=trans,
                zorder=5,
            )
        else:
            ax.scatter(
                station_locations[:, 0],
                station_locations[:, 1],
                c="k",
                s=25,
                marker="^",
                lw=0,
                zorder=5,
                label="Station",
            )

    # xx, yy = _map(xx_mg, yy_mg)
    from matplotlib.colors import LogNorm, SymLogNorm

    if settings["geometry_type"] == "geographic":
        # bp_absmax = np.abs(np.nanmax(beampowers))
        # bp_absmin = np.abs(np.nanmin(beampowers))
        if vmin is None and vmax is None:
            bp_absmax = np.abs(np.nanmax(beampowers))
            vmin = -bp_absmax
            vmax = bp_absmax
        # pcm = ax.pcolormesh(xx, yy, beampowers.T, edgecolors='face', vmin=-bp_absmax, vmax=bp_absmax, transform=trans, norm=SymLogNorm(linthresh=1E-3), cmap='RdBu')
        # pcm = ax.pcolormesh(
        #     xx,
        #     yy,
        #     beampowers.T,
        #     edgecolors="face",
        #     vmin=1e-5,
        #     vmax=vmax,
        #     transform=trans,
        #     norm=LogNorm(),
        #     cmap=cm.batlow,
        # )
        pcm = ax.pcolormesh(
            xx,
            yy,
            beampowers.T,
            edgecolors="face",
            vmin=vmin,
            vmax=vmax,
            transform=trans,
            cmap=cm.vik_r,
        )
    else:
        # force symmetric colorscale
        if vmin is None and vmax is None:
            bp_absmax = np.abs(np.nanmax(beampowers))
            vmin = -bp_absmax
            vmin = 0
            vmax = bp_absmax
        # vmin = -.005
        # vmax = .005
        pcm = ax.pcolormesh(
            xx,
            yy,
            beampowers.T,
            edgecolors="face",
            vmin=vmin,
            vmax=vmax,
            cmap=cm.vik_r,
            shading="nearest",
            # norm=LogNorm(),
        )
        # pcm = ax.contourf(xx, yy, beampowers.T, levels=np.linspace(-1, 1, 100), vmin=vmin, vmax=vmax, cmap=roma_map) # , norm=LogNorm())
        ax.set_xlabel("Distance [km]")
        ax.set_ylabel("Distance [km]")
        # pcm = ax.pcolormesh(xx, yy, beampowers.T, edgecolors='face', vmin=-bp_absmax, vmax=bp_absmax, norm=SymLogNorm(linthresh=1E-3), cmap=roma_map)
    # ax.stock_img()

    x0, y0, w, h = ax.get_position().bounds
    cb_ax = fig.add_axes([x0 + w + 0.025 * w, y0, 0.025 * w, h])
    plt.colorbar(pcm, cax=cb_ax)

    # plot max
    if settings["do_synth"]:
        max_indices = np.unravel_index(
            np.argmax(beampowers, axis=None), beampowers.shape
        )
        lon_max = lons[max_indices[0]]
        lat_max = lats[max_indices[1]]
        if settings["geometry_type"] == "geographic":
            ax.scatter(
                lon_max,
                lat_max,
                facecolors="w",
                edgecolors="k",
                linewidth=0.5,
                s=300,
                marker="*",
                label="Beampower Peak",
                transform=trans,
            )
        elif settings["geometry_type"] == "cartesian":
            ax.scatter(
                lon_max,
                lat_max,
                facecolors="w",
                edgecolors="k",
                linewidth=0.5,
                s=300,
                marker="*",
                label="Beampower Peak",
            )

    if source_loc is not None:
        if settings["geometry_type"] == "geographic":
            pass
            # ax.scatter(
            #     source_loc[0],
            #     source_loc[1],
            #     edgecolors="k",
            #     c="magenta",
            #     linewidth=0.5,
            #     s=50,
            #     marker="*",
            #     transform=trans,
            # )
        else:
            ax.scatter(
                source_loc[0, :],
                source_loc[1, :],
                edgecolors="k",
                c="#e2001a",
                linewidth=0.5,
                s=100,
                marker="*",
                label="Synth. Source",
            )

    # if settings["plot_antipode"]:
    #     max_indices = np.unravel_index(
    #         np.argmax(beampowers.T, axis=None), beampowers.T.shape
    #     )
    #     lon_max = lons[max_indices[0]]
    #     lat_max = lats[max_indices[1]]
    #     logging.info(f"{lon_max=}\t{lat_max=}")
    #     ax.scatter(
    #         lon_max,
    #         lat_max,
    #         edgecolors="k",
    #         linewidth=0.5,
    #         s=10,
    #         marker="*",
    #         transform=ccrs.PlateCarree(),
    #     )

    if title and settings["geometry_type"] == "geographic":
        ax.set_title(title)

    # ax.legend()

    # from cartopy.io import shapereader
    # ax.add_geometries(list(shapereader.Reader('/home/zmaw/u254070/.local/share/cartopy/shapefiles/natural_earth/physical/ne_10m_coastline.shp').geometries()), crs_use)

    # import cartopy.feature as cfeat
    # ax.add_feature(cfeat.GSHHSFeature)

    # ax.gridlines(crs=crs_use)

    # custom lim to check results
    # chile lat -35.47, lon -72.91
    # ax.set_extent([-76, -72, -40, -25], crs=trans)
    # ax.set_ylim(-40, -30)

    if outfile:
        fig.savefig(outfile, dpi=300, transparent=False)
    else:
        plt.show()
    plt.close(fig)

    return pcm
