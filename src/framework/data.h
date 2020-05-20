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
 private:
  bool m_skip_output = false;
  bool m_in_snapshot = false;

 public:
  data_t() {}
  virtual ~data_t() {}

  virtual void init() {}
  void set_skip_output() { m_skip_output = true; }
  bool skip_output() { return m_skip_output; }
  void set_include_in_snapshot() { m_in_snapshot = true; }
  bool in_snapshot() { return m_in_snapshot; }
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
