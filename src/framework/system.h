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
