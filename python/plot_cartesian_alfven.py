#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import cm

cdata = {
    "red": [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.55, 1.0, 1.0), (1.0, 1.0, 1.0),],
    "green": [(0.0, 1.0, 1.0), (0.45, 0.0, 0.0), (0.55, 0.0, 0.0), (1.0, 1.0, 1.0),],
    "blue": [(0.0, 1.0, 1.0), (0.45, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),],
}

hot_cold_cmap = LinearSegmentedColormap("hot_and_cold", cdata, N=1024, gamma=1.0)
plt.register_cmap(cmap=hot_cold_cmap)
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
from datalib import Data
import sys

data = Data(sys.argv[1])


def make_plot(step):
    print("Working on", step)
    time = step * data.conf["dt"] * data.conf["fld_output_interval"]

    data.load_fld(step)
    fig, axes = plt.subplots(6, 1)
    fig.set_size_inches(24.5, 18.5)

    plot_data = [
        data.B3,
        data.Rho_e,
        data.Rho_p,
        (data.E1 * data.B1 + data.E2 * data.B2 + data.E3 * data.B3) / data.B,
        data.gamma_e,
        data.gamma_p,
    ]
    titles = [
        "$B_\\phi$",
        r"$\rho_e$",
        r"$\rho_p$",
        r"$E\cdot B/B$",
        r"$\gamma_e$",
        r"$\gamma_p$",
    ]
    lims = [5e2, 5e3, 5e3, 20.0, 100.0, 100.0]
    ticksize = 22
    labelsize = 30

    for i in range(len(plot_data)):
        if i < 4:
            pmesh = axes[i].pcolormesh(
                data.x1,
                data.x2,
                plot_data[i],
                cmap=hot_cold_cmap,
                vmin=-lims[i],
                vmax=lims[i],
            )
        else:
            pmesh = axes[i].pcolormesh(
                data.x1,
                data.x2,
                plot_data[i],
                cmap=plt.get_cmap('inferno'),
                norm=colors.LogNorm(vmin=1.0, vmax=lims[i]),
            )
        axes[i].contour(
            data.x1, data.x2, data.flux, 20, colors="green", linestyles="solid"
        )
        axes[i].set_aspect("equal")
        axes[i].tick_params(axis="both", labelsize=ticksize)
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cb = plt.colorbar(pmesh, cax=cax)
        cb.ax.tick_params(labelsize=ticksize)
        cb.ax.set_ylabel(titles[i], fontsize=labelsize, rotation=0)
    #     cb.ax.set_title(titles[i], fontsize=labelsize)

    axes[0].text(8, 3.0, f"Time = {time:.2f}", fontsize=labelsize)
    fig.savefig("plots/%05d.png" % step, bbox_inches='tight')
    plt.close(fig)

num_agents = 7

with Pool(processes=num_agents) as pool:
    pool.map(make_plot, data.fld_steps)
