#ifndef __COMPONENT_H_
#define __COMPONENT_H_

#include <string>

namespace Aperture {

class sim_environment;

class data_t {
 public:
  virtual void init(
      const std::string& name, const sim_environment& env) = 0;
};

}

#endif
