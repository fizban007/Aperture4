#ifndef __FRAMEWORK_DATA_H_
#define __FRAMEWORK_DATA_H_

#include <string>

namespace Aperture {

class sim_environment;

class data_t {
 public:
  virtual void init(const std::string& name, const sim_environment& env) {}
  virtual void serialize() {}
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
