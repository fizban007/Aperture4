#ifndef __FRAMEWORK_DATA_H_
#define __FRAMEWORK_DATA_H_

#include <string>

namespace Aperture {

class sim_environment;
class data_exporter;

class data_t {
 public:
  data_t() {}
  virtual ~data_t() {}

  virtual void init() {}

  virtual void data_output(const data_exporter& exporter) {}
  virtual void data_dump(const data_exporter& exporter) {}
};

}  // namespace Aperture

#endif  // __FRAMEWORK_DATA_H_
