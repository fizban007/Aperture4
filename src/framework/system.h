#ifndef __SYSTEM_H_
#define __SYSTEM_H_

#include <cstdint>
#include <set>
#include <string>

namespace Aperture {

class sim_environment;

class system_t {
 public:
  system_t(sim_environment& env) : m_env(env) {}
  virtual ~system_t() {}

  virtual void init() {}
  virtual void register_data_components() {}
  virtual void update(double, uint32_t) {}

 protected:
  sim_environment& m_env;
};

}  // namespace Aperture

#endif
