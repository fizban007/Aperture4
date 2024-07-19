#!/usr/bin/env python3

import h5py
import numpy as np
import toml
from pathlib import Path
import os
import re

extra_fld_keys = ["B", "J", "flux"]

def flag_to_species(flag):
  return flag >> (32 - 4);

class Data:
  _coord_keys = []
  _mesh_loaded = False

  def __init__(self, path):
    self._path = path

    self.reload()

  def __dir__(self):
    return (
      self._fld_keys
      + self._ptc_keys
      + ["load", "load_fld", "load_ptc", "keys", "conf"]
    )

  def __getattr__(self, key):
    if key not in self.__dict__:
      if key in self._fld_keys:
        self.__load_fld_quantity(key)
      elif key in self._ptc_keys:
        self.__load_ptc_quantity(key)
      elif key in self._mesh_keys:
        self.__load_mesh_quantity(key)
      elif key == "keys":
        self.__dict__[key] = self._fld_keys + self._ptc_keys + self._mesh_keys
      elif key == "conf":
        self.__dict__[key] = self._conf
      else:
        return None
    return self.__dict__[key]

  def __load_fld_quantity(self, key):
    path = os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5")
    if key == "flux" and key not in self._original_fld_keys:
      self.__dict__[key] = np.cumsum(self.B1,axis=0)*self.delta[1] - np.cumsum(self.B2,axis=1)*self.delta[0]
    elif key == "B" and key not in self._original_fld_keys:
      self.__dict__[key] = np.sqrt(self.B1 * self.B1 + self.B2 * self.B2 + self.B3 * self.B3)
    elif key == "J" and key not in self._original_fld_keys:
      self.__dict__[key] = np.sqrt(self.J1 * self.J1 + self.J2 * self.J2 + self.J3 * self.J3)
    else:
      data = h5py.File(path, "r")
      self.__dict__[key] = data[key][()]
      data.close()

  def __load_ptc_quantity(self, key):
    path = os.path.join(self._path, f"ptc.{self._current_ptc_step:05d}.h5")
    data = h5py.File(path, "r")
    self.__dict__[key] = data[key][()]
    data.close()

  def __load_mesh_quantity(self, key):
    data = h5py.File(self._meshfile, "r")
    self.__dict__[key] = data[key][()]
    data.close()

  def load(self, step):
    self.load_fld(step)
    self.load_ptc(step)

  def reload(self):
    # reload configuration file
    self._conf = self.load_conf(os.path.join(self._path, "config.toml"))
    self.__load_mesh()

    num_re = re.compile(r"\d+")
    # generate a list of output steps for fields
    self._fld_keys = []
    self.fld_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(self._path).glob("fld.*.h5")
    ]
    if len(self.fld_steps) > 0:
      self.fld_steps.sort()
      self._current_fld_step = self.fld_steps[0]
      f_fld = h5py.File(
        os.path.join(self._path, f"fld.{self._current_fld_step:05d}.h5"),
        "r",
      )
      self._original_fld_keys = list(f_fld.keys())
      self._fld_keys = list(f_fld.keys())
      for k in extra_fld_keys:
        if k not in self._original_fld_keys:
          self._fld_keys.append(k)
      print("fld keys are:", self._fld_keys)
      f_fld.close()

    # generate a list of output steps for particles
    self._ptc_keys = []
    self.ptc_steps = [
      int(num_re.findall(f.stem)[0]) for f in Path(self._path).glob("ptc.*.h5")
    ]
    if len(self.ptc_steps) > 0:
      self.ptc_steps.sort()
      self._current_ptc_step = self.ptc_steps[0]
      f_ptc = h5py.File(
        os.path.join(self._path, f"ptc.{self._current_ptc_step:05d}.h5"),
        "r",
      )
      self._ptc_keys = list(f_ptc.keys())
      print("ptc keys are:", self._ptc_keys)
      f_ptc.close()
  
  def __load_mesh(self):
    print("Base")
    # load mesh file
    if self._mesh_loaded:
      return
    self._meshfile = os.path.join(self._path, "grid.h5")
    f_mesh = h5py.File(os.path.join(self._path, f"grid.h5"), 'r')
    self._mesh_keys = list(f_mesh.keys())
    for k in self._mesh_keys:
      self.__dict__[k] = f_mesh[k][()]
    f_mesh.close()
    # find mesh deltas
    self.delta = np.zeros(len(self._conf["N"]))
    for n in range(len(self.delta)):
      self.delta[n] = self._conf["size"][n] / self._conf["N"][n] * self._conf["downsample"]
    self._mesh_loaded = True


  def load_fld(self, step):
    if not step in self.fld_steps:
      print("Field step not in data directory!")
      return
    self._current_fld_step = step
    for k in self._fld_keys:
      if k in self.__dict__:
        self.__dict__.pop(k, None)
        # self._mesh_loaded = False

  def load_ptc(self, step):
    if not step in self.ptc_steps:
      print("Ptc step not in data directory!")
      return
    self._current_ptc_step = step
    for k in self._ptc_keys:
      if k in self.__dict__:
        self.__dict__.pop(k, None)

  def load_conf(self, path):
    return toml.load(path)

  def particle_series(self, ptc_id, key):
    if isinstance(ptc_id, list) or isinstance(ptc_id, np.ndarray):
      result = np.zeros((len(self.ptc_steps), len(ptc_id)))
      for n in self.ptc_steps:
        self.load_ptc(n)
        for i in ptc_id:
          result[n, i] = self.__getattr__(key)[self.tracked_ptc_id == ptc_id[i]]
    else:
      result = np.zeros(len(self.ptc_steps))
      for n in self.ptc_steps:
        self.load_ptc(n)
        result[n] = self.__getattr__(key)[self.tracked_ptc_id == ptc_id]
    return result

  def time_series_ptc(self, key):
    result = np.zeros(len(self.ptc_steps))
    for n in self.ptc_steps:
      self.load_ptc(n)
      result[n] = self.__getattr__(key)
    return result
    
  def time_series_fld(self, key):
    result = np.zeros(len(self.fld_steps))
    for n in self.fld_steps:
      self.load_fld(n)
      result[n] = self.__getattr__(key)
    return result
