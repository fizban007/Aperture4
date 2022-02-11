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
r = np.sqrt(data.x1**2 + data.x2**2)

def make_plot(step):
    print("Working on", step)
    time = step * data.conf["dt"] * data.conf["fld_output_interval"]

    data.load_fld(step)

    r0 = data.conf['lower'][0]

    Bp = data.conf["Bp"]
    E3 = np.sum(data.E3, axis=0)/data.E3.shape[0]
    # B2 = np.sum((data.B2 - Bp/r)*np.sqrt(r) + Bp, axis=0)/data.B2.shape[0]
    B2 = np.sum(data.B2, axis=0)/data.E3.shape[0]
    # E3vac = np.sum(data_vac.E3, axis=0)/data_vac.E3.shape[0]
    # B2vac = np.sum(data_vac.B2, axis=0)/data_vac.E3.shape[0]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16.0, 9.0)

    tick_size = 25
    label_size = 30

    ax.plot(r[0,:], B2, label="$B_\\theta$")
    ax.plot(r[0,:], E3, label="$E_z$")
    # ax.plot(r_vac[0,:], B2vac, '--', label="$E_z$, FFE")
    # ax.plot(r_vac[0,:], E3vac, '--', label="$B_\\theta$, FFE")
    ax.legend(fontsize=label_size)
    ax.tick_params(labelsize=tick_size)
    ax.set_xlim(max(r0, time + r0 - 15), max(r0 + 16, time + r0 + 1))
    ax.axhline(0.0, c='c', ls='--')
    ax.set_ylim(-3, 12)
    ax.set_xlabel("$r$", fontsize=label_size)
    ax.set_ylabel("$E,B$", fontsize=label_size)

    ax.set_title(f"$t = {time:.1f}$", fontsize=label_size)
    ax.set_position([0.1,0.1,0.8,0.8])
    fig.savefig("plots/%05d.png" % step)
    plt.close(fig)

num_agents = 4

with Pool(processes=num_agents) as pool:
    pool.map(make_plot, data.fld_steps)
