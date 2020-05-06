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
  data = Data("/home/alex/storage/Data/Aperture4/alfven_wave_dw0.2_smaller_Bp/")

  data.load_fld(n)

  flux_lower = 10
  flux_upper = 0.2 * data._conf["Bp"]
  flux_num = 10
  clevel = np.linspace(flux_lower, flux_upper, flux_num)
  color_flux = "#7cfc00"

  fig, axes = plt.subplots(ncols=2,figsize=(13,13))
  ticksize = 15
  titlesize = 25
  Bp = np.sqrt(data.B1 * data.B1 + data.B2 * data.B2)
  b3 = axes[0].pcolormesh(data.x1, data.x2, data.B3 / Bp, cmap=hot_cold_cmap,
                          vmin=-1, vmax=1, shading="gouraud")
  e3 = axes[1].pcolormesh(data.x1, data.x2, data.E3 / Bp, cmap=hot_cold_cmap,
                          vmin=-0.1, vmax=0.1, shading="gouraud")
  axes[0].contour(data.x1, data.x2, data.flux, clevel, colors=color_flux, linewidths=0.6)
  axes[1].contour(data.x1, data.x2, data.flux, clevel, colors=color_flux, linewidths=0.6)
  # axes[0,0].set_title("Explicit", fontsize=titlesize)
  plots = [b3, e3]
  titles = ["$B_\\phi/B_p$", "$E_\\phi/B_p$"]
  for i in range(len(axes)):
    axes[i].set_aspect("equal")
    axes[i].tick_params(axis="both", labelsize=ticksize)
    divider = make_axes_locatable(axes[i])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cb = fig.colorbar(plots[i], orientation="vertical", cax=cax)
    cb.ax.tick_params(labelsize=ticksize)
    cb.ax.set_title(titles[i], fontsize=titlesize)

  axes[1].text(50, 70, "Time = %.1f" % (n * 1.0), fontsize=titlesize)

  fig.savefig("plots/%05d.png" % n)
  plt.close(fig)

# steps_to_plot = data.fld_steps
agents = 7

with Pool(processes=agents) as pool:
    pool.map(make_plot, range(100))
