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
  // By default, data output is "opt-out", so by default all data components
  // will be included in every data output
  void skip_output(bool b) { m_skip_output = b; }
  bool skip_output() { return m_skip_output; }

  // By default, snapshot is "opt-in", so by default any data component will not
  // be included in a snapshot
  void include_in_snapshot(bool b) { m_in_snapshot = b; }
  bool include_in_snapshot() { return m_in_snapshot; }
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
