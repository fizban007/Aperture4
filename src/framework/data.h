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
  bool m_reset_after_output = false;

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

  // Data components can be reset after each output, so that they always track
  // the cumulative amount between two data dumps
  void reset_after_output(bool b) { m_reset_after_output = b; }
  bool reset_after_output() { return m_reset_after_output; }
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
