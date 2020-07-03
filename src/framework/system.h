/*
 * Copyright (c) 2020 Alex Chen.
 * This file is part of Aperture (https://github.com/fizban007/Aperture4.git).
 *
 * Aperture is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Aperture is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __SYSTEM_H_
#define __SYSTEM_H_

#include <cstdint>
#include <set>
#include <string>

namespace Aperture {

class sim_environment;

////////////////////////////////////////////////////////////////////////////////
///  This is the base class for a `system`. A `system` is a module that will be
///  called at every time step to manipulate any given number of data
///  components. It has three main methods that may be implemented: init(),
///  register_data_components(), and update(double, uint32_t).
////////////////////////////////////////////////////////////////////////////////
class system_t {
 public:
  /// Constructor. The `system` has to know about the environment so that it can
  /// register data or get parameters
  system_t(sim_environment& env) : m_env(env) {}
  virtual ~system_t() = default;

  /// Register data components.
  /*!If implemented, this method will be called right after the `system` is
   * constructed, in `sim_environment::register_system<T>()`.
   */
  virtual void register_data_components() {}

  /// Initialize the `system`.
  /*! If implemented, this method will be called together
   * by `sim_environment::init()`. By the time this is called, the data
   * components should have all been registered already.
   */
  virtual void init() {}

  /// Update by a timestep. This method will be called at every timestep
  /*!
   *\param dt    The size of the timestep
   *\param step  The current timestep, useful if the particular module only does
   *             something every few timesteps
   */
  virtual void update(double dt, uint32_t step) {}

 protected:
  /// Keeps a reference to the `sim_environment` so that any derived `system` can
  /// access it via this member
  sim_environment& m_env;
};

}  // namespace Aperture

#endif
