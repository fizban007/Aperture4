#!/usr/bin/env python

import h5py
import numpy as np
import toml
from pathlib import Path
import os
import re


class Data:
  _coord_keys = ["x1", "x2", "r", "theta", "rv", "thetav", "dr", "dtheta"]
  _extra_fld_keys = ["B", "J", "flux"]

  def __init__(self, path):
    self._conf = self.load_conf(os.path.join(path, "config.toml"))
    self._path = path

    # Load mesh file
    self._meshfile = os.path.join(path, "grid.h5")
    f_mesh = h5py.File(os.path.join(self._path, f"grid.h5"), 'r')
    self._mesh_keys = list(f_mesh.keys())
    for k in self._mesh_keys:
      self.__dict__[k] = f_mesh[k][()]
    f_mesh.close()
    # find mesh deltas
    self.delta = np.zeros(len(self._conf["N"]))
    for n in range(len(self.delta)):
      self.delta[n] = self._conf["size"][n] / self._conf["N"][n] * self._conf["downsample"]

    # Generate a list of output steps
    num_re = re.compile(r"\d+")
    self._fld_keys = []
    self.fld_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(path).glob("fld.*.h5")
    ]
    if len(self.fld_steps) > 0:
      self.fld_steps.sort()
      self._current_fld_step = self.fld_steps[0]
      self.fld_interval = self.fld_steps[-1] - self.fld_steps[-2]
      f_fld = h5py.File(
        os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5"),
        "r",
      )
      self._original_fld_keys = list(f_fld.keys())
      self._fld_keys = list(f_fld.keys())
      for k in self._extra_fld_keys:
        if k not in self._original_fld_keys:
          self._fld_keys.append(k)
      print("fld keys are:", self._fld_keys)
      f_fld.close()

    self.ptc_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(path).glob("ptc.*.h5")
    ]
    self._current_fld_step = self.fld_steps[0]
    self._mesh_loaded = False

    if len(self.ptc_steps) > 0:
      self.ptc_steps.sort()
      self.ptc_interval = self.ptc_steps[-1] - self.ptc_steps[-2]
      self._current_ptc_step = self.ptc_steps[0]
      f_ptc = h5py.File(
        os.path.join(self._path, f"ptc.{self._current_ptc_step:05d}.h5"),
        "r",
        swmr=True,
      )
      self._ptc_keys = list(f_ptc.keys())
      f_ptc.close()
    else:
      self.ptc_interval = 0
      self._current_ptc_step = 0
      self._ptc_keys = []
    self.__dict__.update(("_" + k, None) for k in (self._fld_keys + self._ptc_keys))

  def __dir__(self):
    return (
      self._fld_keys
      + self._ptc_keys
      + self._coord_keys
      + ["load", "load_fld", "load_ptc", "keys", "conf"]
    )

  def __getattr__(self, key):
    if key in (self._fld_keys + self._ptc_keys + self._coord_keys):
      content = getattr(self, "_" + key)
      if content is not None:
        return content
      else:
        self._load_content(key)
        return getattr(self, "_" + key)
    elif key == "keys":
      return self._fld_keys + self._ptc_keys
    elif key == "conf":
      return self._conf
    else:
      return None

  def load(self, step):
    self.load_fld(step)
    self.load_ptc(step)

  def load_fld(self, step):
    if not step in self.fld_steps:
      print("Field step not in data directory!")
      return
    self._current_fld_step = step
    for k in self._fld_keys:
      if k not in Data._coord_keys:
        setattr(self, "_" + k, None)
        # self._mesh_loaded = False

  def load_ptc(self, step):
    if not step in self.ptc_steps:
      print("Ptc step not in data directory!")
      return
    self._current_ptc_step = step
    for k in self._ptc_keys:
      setattr(self, "_" + k, None)

  def _load_content(self, key):
    if key in self._fld_keys:
      self._load_fld(key)
    elif key in self._ptc_keys:
      self._load_ptc(key)
    elif key in self._coord_keys:
      self._load_mesh()

  def _load_fld(self, key):
    path = os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5")
    data = h5py.File(path, "r", swmr=True)
    if key == "flux":
      self._load_mesh()
      dtheta = (
        self._theta[self._conf["guard"][1] + 2]
        - self._theta[self._conf["guard"][1] + 1]
      )
      self._flux = np.cumsum(
        self.B1 * self._rv * self._rv * np.sin(self._thetav) * dtheta, axis=0
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
      setattr(self, "_" + key, data[key][()])
      data.close()

  def _load_ptc(self, key):
    path = os.path.join(self._path, f"ptc.{self._current_ptc_step:05d}.h5")
    data = h5py.File(path, "r", swmr=True)
    setattr(self, "_" + key, data[key][()])
    data.close()

  def _load_mesh(self):
    if self._mesh_loaded:
      return
    meshfile = h5py.File(self._meshfile, "r", swmr=True)

    self._x1 = meshfile["x1"][()]
    self._x2 = meshfile["x2"][()]
    # self._r = np.pad(
    #     np.exp(
    #         np.linspace(
    #             0,
    #             self._conf["Grid"]["size"][0],
    #             self._conf["Grid"]["N"][0]
    #             // self._conf["Simulation"]["downsample"],
    #         )
    #         + self._conf["Grid"]["lower"][0]
    #     ),
    #     self._conf["Grid"]["guard"][0],
    #     "constant",
    # )
    # self._theta = np.pad(
    #     np.linspace(
    #         0,
    #         self._conf["Grid"]["size"][1],
    #         self._conf["Grid"]["N"][1] // self._conf["Simulation"]["downsample"],
    #     )
    #     + self._conf["Grid"]["lower"][1],
    #     self._conf["Grid"]["guard"][1],
    #     "constant",
    # )
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

    meshfile.close()
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

  def load_conf(self, path):
    return toml.load(path)
