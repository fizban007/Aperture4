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

  /// Initialize the data component.
  /*!
   * Similar to systems, this will be called by the init() method in
   * sim_environment. This will reset the data to a "zero" state, whenever it is
   * appropriately defined. This method should not handle resource allocation,
   * which should happen in the constructor. This method will be called whenever
   * a data component needs to be reset.
   */
  virtual void init() {}

  // By default, data output is "opt-out", so by default all data components
  // will be included in every data output

  /// Set wether this data component will skip the data output.
  /*!
   * Normally all data components will participate in data output at a given
   * interval, unless it is "opted-out" by calling
   *
   *     data.skip_output(true);
   *
   * This is useful for some intermediate data one needs to keep track of, like
   * the rng states.
   */
  void skip_output(bool b) { m_skip_output = b; }

  /// Get whether this data component will skip the output.
  bool skip_output() { return m_skip_output; }

  // By default, snapshot is "opt-in", so by default any data component will not
  // be included in a snapshot

  /// Set whether this data component will be included in a snapshot.
  /*!
   * Normally none of the data components will participate in a snapshot unless
   * explicitly specified. This is useful to reduce the size of the snapshot
   * file. To "opt-in", explicitly call this:
   *
   *     data.include_in_snapshot(true);
   */
  void include_in_snapshot(bool b) { m_in_snapshot = b; }

  /// Get whether this data component will be included in a snapshot.
  bool include_in_snapshot() { return m_in_snapshot; }

  // Data components can be reset after each output, so that they always track
  // the cumulative amount between two data dumps

  /// Mark whether this data component should be reset after each data output.
  /*!
   * This is useful if we want to accumulate certain quantities over two data
   * outputs, or average over this time interval. By default this is not done.
   * To "opt-in", explicitly call this:
   *
   *     data.reset_after_output(true);
   */
  void reset_after_output(bool b) { m_reset_after_output = b; }

  /// Get wether this data component should be reset after output.
  bool reset_after_output() { return m_reset_after_output; }
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
