#ifndef __FRAMEWORK_DATA_H_
#define __FRAMEWORK_DATA_H_

#include <string>

namespace Aperture {

class sim_environment;

////////////////////////////////////////////////////////////////////////////////
///  A very basic interface class to inherit from. All data components are
///  stored as an implementation to this interpace.
////////////////////////////////////////////////////////////////////////////////
class data_t {
 public:
  data_t() {}
  virtual ~data_t() {}

  virtual void init() {}
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
