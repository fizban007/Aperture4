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
    sinth = data.conf["muB"]
    costh = np.sqrt(1.0 - sinth * sinth)

    data.load_fld(step)
    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(20.5, 26.5)

    plot_data = [
        [data.B3, data.Rho_e],
        [
            data.Rho_p,
            # (data.E1 * data.B1 + data.E2 * data.B2 + data.E3 * data.B3) / data.B,
            data.E1 * sinth + data.E2 * costh
        ],
        [data.gamma_e, data.pair_produced],
        # [data.gamma_e, data.gamma_p],
    ]
    titles = [
        [r"$B_z$", r"$\rho_e$"], [r"$\rho_p$",
        # r"$E\cdot B/B$"], [r"$\gamma_e$", r"Pair Produced"],
        # r"$E\cdot B_0/B_0$"], [r"$\gamma_e$", r"$\gamma_p$"],
        r"$E\cdot B_0/B_0$"], [r"$\gamma_e$", r"Pair Produced"],
    ]
    lims = [[2e3, 5e4], [5e4, 20.0], [100.0, 100.0]]
    ticksize = 22
    labelsize = 30

    for j in range(len(plot_data)):
        for i in range(len(plot_data[j])):
            # print(titles[j][i], plot_data[j][i].shape)
            lin_idx = i + j * len(plot_data[j])
            if lin_idx < 4:
                pmesh = axes[j][i].pcolormesh(
                    data.x1,
                    data.x2,
                    plot_data[j][i],
                    cmap=hot_cold_cmap,
                    vmin=-lims[j][i],
                    vmax=lims[j][i],
                )
            else:
              if lin_idx == 4:
                pmesh = axes[j][i].pcolormesh(
                    data.x1,
                    data.x2,
                    plot_data[j][i],
                    cmap=plt.get_cmap("inferno"),
                    norm=colors.LogNorm(vmin=1.0, vmax=lims[j][i]),
                )
              else:
                pmesh = axes[j][i].pcolormesh(
                    data.x1,
                    data.x2,
                    plot_data[j][i],
                    cmap=plt.get_cmap("inferno"),
                    vmin=0,
                    vmax=0.2
                )

            axes[j][i].contour(
                data.x1, data.x2, data.flux, 20, colors="green", linestyles="solid", linewidths=0.5
            )
            axes[j][i].set_aspect("equal")
            axes[j][i].tick_params(axis="both", labelsize=ticksize)
            divider = make_axes_locatable(axes[j][i])
            cax = divider.append_axes("right", size="2%", pad=0.05)
            cb = plt.colorbar(pmesh, cax=cax)
            cb.ax.tick_params(labelsize=ticksize)
            cb.ax.set_ylabel(titles[j][i], fontsize=labelsize, rotation=270, labelpad=25)
    #     cb.ax.set_title(titles[i], fontsize=labelsize)

    axes[0][0].text(0, 11.0, f"Time = {time:.2f}", fontsize=labelsize)
    fig.savefig("plots/%05d.png" % step, bbox_inches="tight")
    plt.close(fig)


num_agents = 7

with Pool(processes=num_agents) as pool:
    pool.map(make_plot, data.fld_steps)
