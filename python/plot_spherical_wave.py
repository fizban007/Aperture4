#!/usr/bin/env python3

from datalib_logsph import Data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge, Arc, Rectangle
from matplotlib import cm
from multiprocessing import Pool

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
import numpy as np


def make_plot(n):
  print("Working on %d" % n)
  data_expl = Data("/home/alex/storage/Data/Tests/spherical_wave_expl_k10.0/")
  data_impl = Data("/home/alex/storage/Data/Tests/spherical_wave_impl_k10.0/")

  data_expl.load_fld(n)
  data_impl.load_fld(n)

  fig, axes = plt.subplots(ncols=2,figsize=(13,13))
  ticksize = 15
  titlesize = 25

  expl = axes[0].pcolormesh(data_expl.x1, data_expl.x2, data_expl.B3, cmap=hot_cold_cmap,
                          vmin=-1, vmax=1, shading="gouraud")
  impl = axes[1].pcolormesh(data_impl.x1, data_impl.x2, data_impl.B3, cmap=hot_cold_cmap,
                          vmin=-1, vmax=1, shading="gouraud")
  axes[0].set_title("Explicit", fontsize=titlesize)
  axes[1].set_title("Semi-implicit", fontsize=titlesize)
  for ax in axes:
    ax.set_aspect("equal")
    ax.tick_params(axis="both", labelsize=ticksize)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(impl, cax=cax)
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_title("$B_\\phi$", fontsize=titlesize)
  axes[1].text(20, 31, "Time = %.1f" % (n * 1.0), fontsize=titlesize)

  fig.savefig("plots/%05d.png" % n)
  plt.close(fig)

# steps_to_plot = data.fld_steps
agents = 7

with Pool(processes=agents) as pool:
    pool.map(make_plot, range(100))
