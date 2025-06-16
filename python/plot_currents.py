import matplotlib
# matplotlib.rc("text", usetex=True)
# matplotlib.rc("font", family="serif")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py
import sys
import os
aperture_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append('.')
sys.path.append(os.path.join(aperture_dir, "python"))
from datalib import Data, flag_to_species

import pplotutil as pu

plt.style.use(os.path.join(aperture_dir, "python", "aa_ppt.mplstyle"))

def plot_currents(*args, **kwargs):
    step = int(args[0])

    data = Data("./")
    print(data.conf)
    print(data.fld_steps[-1])

    data.load_fld(step)
    print("data has up to", data.fld_steps[-1], "steps")
    print("showing data step", step)

    #================================================================
    # Simulation parameters
    Lx, Ly = data.conf["size"]
    Nx, Ny = data.conf["N"]
    rfac = data.conf["downsample"]
    Nxr, Nyr = Nx // rfac, Ny // rfac
    xmin, ymin = data.conf["lower"]
    xmax, ymax = xmin + Lx, ymin + Ly

    try:
        sigma = data.conf["sigma"]
    except:
        sigma = data.conf["sigma_cs"]
    B0 = np.sqrt(sigma)
    betad = data.conf["current_sheet_drift"]
    delta = B0 / (2.0 * betad)
    J0 = B0 / delta

    dt = data.conf["dt"]
    fdump = data.conf["fld_output_interval"]

    xlabel = r"$x/d_e$"
    ylabel = r"$y/d_e$"
    xnorm = 1.0
    ynorm = 1.0

    x = data.x1  # 1D x-array is x[0,:]
    y = data.x2  # 1D y-array is y[:,0]

    # x_periodic = data.conf["periodic_boundary"][0]
    # y_periodic = data.conf["periodic_boundary"][1]
    x_periodic = False
    y_periodic = False

    ix_bottom = 0
    iy_bottom = 0
    if "damping_length" in data.conf:
        if not y_periodic:
            iy_bottom = data.conf["damping_length"] // rfac
        if not x_periodic:
            ix_bottom = data.conf["damping_length"] // rfac
        print("pml at distance %g from box bottom" % (y[iy_bottom,0] - y[0,0]))
        print("pml at distance %g from box side" % (x[0,ix_bottom] - x[0,0]))

    #================================================================
    # Arguments
    xmin_plot = xmin / xnorm
    if "xmin" in kwargs:
        xmin_plot = kwargs["xmin"]
    xmax_plot = xmax / xnorm
    if "xmax" in kwargs:
        xmax_plot = kwargs["xmax"]

    ymin_plot = ymin / ynorm
    if "ymin" in kwargs:
        ymin_plot = kwargs["ymin"]
    ymax_plot = ymax / ynorm
    if "ymax" in kwargs:
        ymax_plot = kwargs["ymax"]

    vmin = None
    if "vmin" in kwargs:
        vmin = kwargs["vmin"]
    vmax = None
    if "vmax" in kwargs:
        vmax = kwargs["vmax"]

    cmap = "RdBu_r"
    if "cmap" in kwargs:
        cmap = kwargs["cmap"]
    clog = False
    if "clog" in kwargs:
        clog = kwargs["clog"]
    linthresh = None
    if "linthresh" in kwargs:
        linthresh = kwargs["linthresh"]
    linscale = 1.0
    if "linscale" in kwargs:
        linscale = kwargs["linscale"]

    levels = 10
    if "levels" in kwargs:
        levels = kwargs["levels"]

    allowed_az0_policies = ["lower", "extremum"]
    az0_policy = "lower"
    if "az0_policy" in kwargs:
        az0_policy = kwargs["az0_policy"]

    if az0_policy not in allowed_az0_policies:
        print("kwarg 'az0_policy' must be one of:", allowed_az0_policies)
        print("Got", az0_policy)
        print("Aborting...")
        sys.exit(0)

    showsep = False
    if "showsep" in kwargs:
        showsep = kwargs["showsep"]

    nrows = 1
    if "nrows" in kwargs:
        nrows = kwargs["nrows"]

    ncols = 4
    if "ncols" in kwargs:
        ncols = kwargs["ncols"]

    if nrows * ncols != 4:
        print("Error, got (nrows, ncols) = (%d, %d)" % (nrows, ncols))
        print("But need nrows * ncols == 4.")
        print("Aborting...")
        sys.exit(0)

    #================================================================
    # Calculations on fields

    Bx = data.B1
    By = data.B2
    xAz = x[iy_bottom:-iy_bottom-1,ix_bottom:-ix_bottom-1]
    yAz = y[iy_bottom:-iy_bottom-1,ix_bottom:-ix_bottom-1]
    BxAz = Bx[iy_bottom:-iy_bottom-1,ix_bottom:-ix_bottom-1]
    ByAz = By[iy_bottom:-iy_bottom-1,ix_bottom:-ix_bottom-1]
    Az = pu.calc_Az(xAz[0,:], yAz[:,0], BxAz.T, ByAz.T).T / (B0 * Lx)
    if az0_policy == "lower":
        Az0 = Az[0,Nxr//2]
    elif az0_policy == "extremum":
        Az0 = np.max(Az)
    Az = (Az - Az0)
    print("min(Az / B0 Lx), max(Az / B0 Lx) = %g, %g" % 
        (np.min(Az), np.max(Az)))
    if levels > 0:
        levels = np.linspace(-1.0 * Ly/Lx, 1.0 * Ly/Lx, num=4*levels)
    else:
        levels = None

    ix0 = np.argmin(np.abs(x[0,:] - (xmax + xmin)/2))
    iy0 = np.argmin(np.abs(y[:,0] - (ymax + ymin)/2))
    Azsep = Az[iy0, ix0]
    print("Az / B0 Lx at box midpoint is " + str(Azsep))

    #================================================================
    # Set up figure

    itpad = "{:>6}".format(step*fdump)
    timestr = "timestep="+itpad+r"$; ct/L$=%.2f" \
        % (step*dt*fdump/Lx)
    titlestr = timestr
    # tick_size = 25
    # label_size = 30

    pad = 0.05
    delta = pad * min(xmax_plot - xmin_plot, ymax_plot - ymin_plot)
    textx = xmax_plot - delta
    texty = ymax_plot - delta
    textha = "right"
    textva = "top"
    lw = plt.rcParams["lines.linewidth"]
    path_effects = [
        pe.Stroke(linewidth = 1.5*lw, foreground = 'w', alpha=1.0),
        pe.Normal(),
    ]

    norm = colors.Normalize(vmin, vmax)
    if clog:
        if linthresh is None:
            norm = colors.LogNorm(vmin, vmax)
        else:
            norm = colors.SymLogNorm(
                linthresh,
                vmin=vmin, vmax=vmax,
                linscale=linscale,
            )

    cbar_size = "5%"
    cbar_mode = "single"
    cbar_pad = 0.3
    cbar_location = "top"
    if nrows >= 2:
        cbar_size = "1%"
        cbar_mode = "single"
        cbar_pad = 0.0
        cbar_location = "right"

    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure()
    fig.patch.set_facecolor('w')
    grid = ImageGrid(
        fig, 111,
        nrows_ncols = (nrows, ncols),
        axes_pad = 0.0,
        cbar_location = cbar_location,
        cbar_size = cbar_size,
        cbar_mode = cbar_mode,
        cbar_pad = cbar_pad,
    )

    #================================================================
    # Jx
    ax = grid[0]
    ax.grid(False)
    ax.set_title(titlestr, loc = "left")
    pm = ax.pcolormesh(
        x / xnorm, y / ynorm, data.J1 / J0,
        norm=norm,
        shading='gouraud',
        cmap=cmap,
    )
    if levels is not None:
        contours = ax.contour(
            xAz / xnorm, yAz / ynorm, Az,
            levels = levels,
            colors = "black",
        ).levels
        print("Plotting contour levels: ", contours)
        if showsep:
            csep = ax.contour(
                xAz / xnorm, yAz / ynorm, Az,
                levels = [Azsep],
                colors = "black",
            )
            csep.set(path_effects = path_effects)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((xmin_plot, xmax_plot))
    ax.set_ylim((ymin_plot, ymax_plot))
    ax.set_aspect('equal')
    ax.text(
        textx, texty, r"$J_x/B_0$",
        ha = textha, va = textva,
        path_effects = path_effects,
    )

    #================================================================
    # Jy
    ax = grid[1]
    ax.grid(False)
    # ax.set_title(titlestr, loc = "left")
    pm = ax.pcolormesh(
        x / xnorm, y / ynorm, data.J2 / J0,
        norm=norm,
        shading='gouraud',
        cmap=cmap,
    )
    if levels is not None:
        contours = ax.contour(
            xAz / xnorm, yAz / ynorm, Az,
            levels = levels,
            colors = "black",
        ).levels
        if showsep:
            csep = ax.contour(
                xAz / xnorm, yAz / ynorm, Az,
                levels = [Azsep],
                colors = "black",
            )
            csep.set(path_effects = path_effects)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((xmin_plot, xmax_plot))
    ax.set_ylim((ymin_plot, ymax_plot))
    ax.set_aspect('equal')
    ax.text(
        textx, texty, r"$J_y/B_0$",
        ha = textha, va = textva,
        path_effects = path_effects,
    )

    #================================================================
    # Jz
    ax = grid[2]
    ax.grid(False)
    # ax.set_title(titlestr, loc = "left")
    pm = ax.pcolormesh(
        x / xnorm, y / ynorm, data.J3 / B0,
        norm=norm,
        shading='gouraud',
        cmap=cmap,
    )
    if levels is not None:
        contours = ax.contour(
            xAz / xnorm, yAz / ynorm, Az,
            levels = levels,
            colors = "black",
        ).levels
        if showsep:
            csep = ax.contour(
                xAz / xnorm, yAz / ynorm, Az,
                levels = [Azsep],
                colors = "black",
            )
            csep.set(path_effects = path_effects)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((xmin_plot, xmax_plot))
    ax.set_ylim((ymin_plot, ymax_plot))
    ax.set_aspect('equal')
    ax.text(
        textx, texty, r"$J_z/B_0$",
        ha = textha, va = textva,
        path_effects = path_effects,
    )

    #================================================================
    # rho
    ax = grid[3]
    ax.grid(False)
    # ax.set_title(titlestr, loc = "left")
    pm = ax.pcolormesh(
        x / xnorm, y / ynorm, (data.Rho_p + data.Rho_e) / J0,
        norm=norm,
        shading='gouraud',
        cmap=cmap,
    )
    if levels is not None:
        contours = ax.contour(
            xAz / xnorm, yAz / ynorm, Az,
            levels = levels,
            colors = "black",
        ).levels
        if showsep:
            csep = ax.contour(
                xAz / xnorm, yAz / ynorm, Az,
                levels = [Azsep],
                colors = "black",
            )
            csep.set(path_effects = path_effects)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((xmin_plot, xmax_plot))
    ax.set_ylim((ymin_plot, ymax_plot))
    ax.set_aspect('equal')
    ax.text(
        textx, texty, r"$(n_p - n_e)$",
        ha = textha, va = textva,
        path_effects = path_effects,
    )

    #================================================================
    # Colorbar
    ax.cax.grid(False)
    ax.cax.colorbar(pm)

    fname = "./plots/currents_%s.png" % str(int(step))
    if "save" in kwargs:
        fname = kwargs["save"]
    print("Saving figure: ", fname)
    plt.savefig(fname)

if __name__ == "__main__":
    args, kwargs = pu.getArgsAndKwargs(sys.argv[1:])
    if len(args) == 0:
        print("python plot_currents.py [stepnum]")
        print("  IMPORTANT: Run from the Data directory of a simulation.")
        print("  Plot plasma number density at 'timestep'")
        print("  vmin/vmax = the min/max colorbar values.")
        print("  xmin/xmax/ymin/ymax = the min/max x/y-values")
        print("  cmap = RdBu_r(default)")
        print("  clog = True/False(default)")
        print("  lin[thresh|scale] = default [None|1.0]")
        print("     if clog is True, and linthresh is NOT None, use a")
        print("     symlog colorscale with lin[thresh|scale] passed to")
        print("     matplotlib's [linthresh|linscale] parameters")
        print("  levels = (num contour levels)")
        print("  nrows,ncols = 1,4(default) -- num rows,cols in plot")
        print("  az0_policy = [lower(default)|extremum]")
        print("    Where to set Az = 0")
        print("    'lower' - At the bottom lower end of box (outside of PML)")
        print("    'extremum' - Set zero at deepest plasmoid-buried flux.")
        # print("  clog = False/True(default) -- use a log colorscale")
        sys.exit(0)
    plot_currents(*args, **kwargs)
