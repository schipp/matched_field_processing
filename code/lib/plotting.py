# only plot stack, if something to stack
# from sven_utils import suColors
# my_cmap = suColors.get_custom_cmap(name='devon', inverted=True)
# if len(beampowers_per_start_time) > 1:
import logging

from cmcrameri import cm


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
    import cartopy.crs as ccrs
    import numpy as np
    import pylab as plt

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
    # _map.drawmeridians(np.arange(0.,420.,60.))
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
            cmap=cm.batlow,
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
            cmap=cm.batlow,
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

    if settings["plot_antipode"]:
        max_indices = np.unravel_index(
            np.argmax(beampowers.T, axis=None), beampowers.T.shape
        )
        lon_max = lons[max_indices[0]]
        lat_max = lats[max_indices[1]]
        logging.info(f"{lon_max=}\t{lat_max=}")
        ax.scatter(
            lon_max,
            lat_max,
            edgecolors="k",
            linewidth=0.5,
            s=10,
            marker="*",
            transform=ccrs.PlateCarree(),
        )

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
