#ifndef __SYSTEM_H_
#define __SYSTEM_H_

namespace Aperture {

class sim_environment;
class data_store_t;

class system_t {
 public:
  void init_system(sim_environment& env) {
    m_env = &env;
    init();
    register_callbacks(env);
  }

  virtual void register_dependencies(sim_environment&) = 0;
  virtual void init() = 0;
  virtual void update(double) = 0;
  virtual void destroy() = 0;

  virtual void register_callbacks(sim_environment&) = 0;

 protected:
  sim_environment* m_env;
};

}

#endif
