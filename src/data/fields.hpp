#ifndef __FIELDS_H_
#define __FIELDS_H_

#include "framework/data.h"

namespace Aperture {

template <typename Conf>
class vector_field : public data_t {
 public:
  typename Conf::multi_array_t v1;
  typename Conf::multi_array_t v2;
  typename Conf::multi_array_t v3;
  const Conf& m_conf;

 public:
  vector_field(const Conf& conf) : m_conf(conf) {}

  void init(const sim_environment& env);
};

template <typename Conf>
class scalar_field : public data_t {
 public:
  typename Conf::multi_array_t v;
  const Conf& m_conf;

 public:
  scalar_field(const Conf& conf) : m_conf(conf) {}

  void init(const sim_environment& env);
};

}

#endif // __FIELDS_H_
