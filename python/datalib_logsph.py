#!/usr/bin/env python

import h5py
import numpy as np
import toml
from pathlib import Path
import os
import re
from datalib import Data, extra_fld_keys, flag_to_species

class DataSph(Data):
  _coord_keys = ["r", "theta", "rv", "thetav", "dr", "dtheta"]
  _mesh_loaded = False

  def __init__(self, path):
    self._path = path
    self.reload()
    self.__load_sph_mesh()

  def __load_fld_quantity(self, key):
    path = os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5")
    if key == "flux":
      self.__load_sph_mesh()
      self._flux = np.cumsum(
        self.B1 * self._rv * self._rv * np.sin(self._thetav) * self._dtheta, axis=0
      )
    elif key == "B":
      self._B = np.sqrt(self.B1 * self.B1 + self.B2 * self.B2 + self.B3 * self.B3)
    elif key == "J":
      self._J = np.sqrt(self.J1 * self.J1 + self.J2 * self.J2 + self.J3 * self.J3)
    elif key == "EdotB":
      self._EdotB = self.E1 * self.B1 + self.E2 * self.B2 + self.E3 * self.B3
    elif key == "JdotB":
      self._JdotB = self.J1 * self.B1 + self.J2 * self.B2 + self.J3 * self.B3
      # elif key == "EdotB":
      #     setattr(self, "_" + key, data["EdotBavg"][()])
    else:
      data = h5py.File(path, "r", swmr=True)
      setattr(self, "_" + key, data[key][()])
      data.close()

  def __load_sph_mesh(self):
    print("Derived")
    # if self._mesh_loaded:
    #   return
    self._meshfile = os.path.join(self._path, "grid.h5")
    f_mesh = h5py.File(self._meshfile, "r", swmr=True)

    self._mesh_keys = list(f_mesh.keys())
    for k in self._mesh_keys:
      self.__dict__[k] = f_mesh[k][()]
    f_mesh.close()
    
    self._r = np.exp(
      np.linspace(
        0,
        self._conf["size"][0],
        self._conf["N"][0]
        // self._conf["downsample"],
      ) + self._conf["lower"][0]
    )
    self._theta = np.linspace(
      0,
      self._conf["size"][1],
      self._conf["N"][1] // self._conf["downsample"],
    ) + self._conf["lower"][1]

    self._rv, self._thetav = np.meshgrid(self._r, self._theta)
    self._dr = (
      self._r[self._conf["guard"][0] + 2]
      - self._r[self._conf["guard"][0] + 1]
    )
    self._dtheta = (
      self._theta[self._conf["guard"][1] + 2]
      - self._theta[self._conf["guard"][1] + 1]
    )

    self._mesh_loaded = True
