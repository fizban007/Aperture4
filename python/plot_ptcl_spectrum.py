import matplotlib
# matplotlib.rc("text", usetex=True)
# matplotlib.rc("font", family="serif")
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import sys
import os
aperture_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.append('.')
sys.path.append(os.path.join(aperture_dir, "python"))
from datalib import Data, flag_to_species

import pplotutil as pu

plt.style.use(os.path.join(aperture_dir, "python", "aa_ppt.mplstyle"))

def plot_ptcl_spectrum(*args, **kwargs):
    step = int(args[0])

    xmin_plot = None
    if "xmin" in kwargs:
        xmin_plot = kwargs["xmin"]
    xmax_plot = None
    if "xmax" in kwargs:
        xmax_plot = kwargs["xmax"]

    ymin_plot = None
    if "ymin" in kwargs:
        ymin_plot = kwargs["ymin"]
    ymax_plot = None
    if "ymax" in kwargs:
        ymax_plot = kwargs["ymax"]

    vmin = 0.1
    if "vmin" in kwargs:
        vmin = kwargs["vmin"]
    vmax = 10.0
    if "vmax" in kwargs:
        vmax = kwargs["vmax"]

    cmap = "inferno"
    if "cmap" in kwargs:
        cmap = kwargs["cmap"]
    clog = True
    if "clog" in kwargs:
        clog = kwargs["clog"]

    norm = "one"
    if "norm" in kwargs:
        norm = kwargs["norm"]

    allowed_norms = ["one", "ncr"]
    if norm not in allowed_norms:
        print("kwarg 'norm' must be one of:", allowed_norms)
        print("Got:", norm)
        print("Aborting...")
        sys.exit(0)
    cnorm = norm

    allowed_az0_policies = ["lower", "extremum"]
    az0_policy = "lower"
    if "az0_policy" in kwargs:
        az0_policy = kwargs["az0_policy"]

    if az0_policy not in allowed_az0_policies:
        print("kwarg 'az0_policy' must be one of:", allowed_az0_policies)
        print("Got", az0_policy)
        print("Aborting...")
        sys.exit(0)

    levels = 30
    if "levels" in kwargs:
        levels = kwargs["levels"]

    data = Data("./")
    print(data.conf)
    print(data.fld_steps[-1])

    data.load_fld(step)
    print("data has up to", data.fld_steps[-1], "steps")
    print("showing data step", step)

    # tick_size = 25
    # label_size = 30
    #================================================================
    # Simulation parameters
    Lx, Ly = data.conf["size"]
    Nx, Ny = data.conf["N"]
    rfac = data.conf["downsample"]
    Nxr, Nyr = Nx // rfac, Ny // rfac
    xmin, ymin = data.conf["lower"]
    xmax, ymax = xmin + Lx, ymin + Ly

    dt = data.conf["dt"]
    fdump = data.conf["fld_output_interval"]

    try:
        sigma = data.conf["sigma"]
    except:
        sigma = data.conf["sigma_cs"]
    B0 = np.sqrt(sigma)

    betad = data.conf["current_sheet_drift"]
    delta = B0 / (2 * betad)
    ncr = B0 / delta

    x_periodic = data.conf["periodic_boundary"][0]
    y_periodic = data.conf["periodic_boundary"][1]
    # x_periodic = False
    # y_periodic = False

    x = data.x1  # 1D x-array is x[0,:]
    y = data.x2  # 1D y-array is y[:,0]
    ix_bottom = 0
    iy_bottom = 0
    if "damping_length" in data.conf:
        if not y_periodic:
            iy_bottom = data.conf["damping_length"] // rfac
        if not x_periodic:
            ix_bottom = data.conf["damping_length"] // rfac
        print("pml at distance %g from box bottom" % (y[iy_bottom,0] - y[0,0]))
        print("pml at distance %g from box side" % (x[0,ix_bottom] - x[0,0]))

    itpad = "{:>6}".format(step*fdump)
    timestr = "timestep="+itpad+r"$; ct/L$=%.2f" \
        % (step*dt*fdump/Lx)
    titlestr = timestr

    fig, ax = plt.subplots(1, 1)
    ax.grid(False)
    # fig.set_size_inches(16.0, 10.0)
    fig.patch.set_facecolor('w')

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
    rho = data.Rho_p - data.Rho_e
    # print(rho.shape, x.shape)

    rhonorm = 1.0
    clabel = r'$(n_e + n_p)$'
    if cnorm == "ncr":
        rhonorm = nrc
        clabel = r'$(n_e + n_p)/n_{\rm cr}$'

    cmap = plt.get_cmap(cmap)
    cmap.set_bad('k')

    norm = colors.Normalize(vmin, vmax)
    if clog:
        norm = colors.LogNorm(vmin, vmax)
    mesh_rho = ax.pcolormesh(
        x, y, rho / rhonorm,
        norm=norm,
        shading='gouraud',
        cmap=cmap,
    )
    if levels is not None:
        contours = ax.contour(
            xAz, yAz, Az,
            levels = levels,
            colors = "black",
        ).levels
        print("Plotting contour levels: ", contours)
    ax.set_title(titlestr, loc = "left")
    # ax.tick_params(labelsize=tick_size)
    ax.set_xlabel(r'$x/d_e$')
    ax.set_ylabel(r'$y/d_e$')
    ax.set_xlim((xmin_plot, xmax_plot))
    ax.set_ylim((ymin_plot, ymax_plot))
    ax.set_aspect('equal')
    # ax.set_ylim(0.0, 0.5 * data.conf['size'][1])
    yhalf = 0.0
    ytop  = 0.5 * data.conf['size'][1]
    # ax.set_ylim(yhalf + 0.25 * (ytop - yhalf), ytop - 0.25 * (ytop - yhalf))

    divider1 = make_axes_locatable(ax)
    # cax1 = divider1.append_axes("right", size="3%", pad=0.05)
    # cb1 = plt.colorbar(mesh_rho, cax=cax1)
    cb1 = plt.colorbar(
        mesh_rho,
        ax=ax,
        location = "right",
        pad = 0.05,
        fraction = 0.10,
        # orientation = "horizontal",
    )
    # cb1.ax.tick_params(labelsize=tick_size)
    # cb1.ax.set_ylabel('$n_e + n_p$', fontsize=label_size, rotation=270, labelpad=20)
    # cb1.ax.set_ylabel('$(n_e + n_p)/n_0$', rotation=270, labelpad=20)
    cb1.ax.set_ylabel(clabel, rotation=270, labelpad=20)
    print("max density is", np.max(data.Rho_p - data.Rho_e))

    fname = "./plots/density_%s.png" % str(int(step))
    if "save" in kwargs:
        fname = kwargs["save"]
    print("Saving figure: ", fname)
    plt.savefig(fname)

if __name__ == "__main__":
    args, kwargs = pu.getArgsAndKwargs(sys.argv[1:])
    if len(args) == 0:
        print("python plot_ptcl_spectrum.py [stepnum]")
        print("  IMPORTANT: Run from the Data directory of a simulation.")
        print("  Plot plasma number density at 'timestep'")
        print("  vmin/vmax = the min/max colorbar values.")
        print("  xmin/xmax/ymin/ymax = the min/max x/y-values")
        print("  cmap = inferno(default)")
        print("  clog = False/True(default) -- use a log colorscale")
        print("  levels = (num contour levels)")
        print("  norm = [one(default)|ncr]")
        print("    How to normalize the colorbar")
        print("    'one' - Normalize density to 1 (default simulation units)")
        print("    'ncr' - Critical density needed to supply initial current")
        print("  az0_policy = [lower(default)|extremum]")
        print("    Where to set Az = 0")
        print("    'lower' - At the bottom lower end of box (outside of PML)")
        print("    'extremum' - Set zero at deepest plasmoid-buried flux.")
        sys.exit(0)
    plot_density(*args, **kwargs)
