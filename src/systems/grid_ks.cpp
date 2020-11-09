#include "grid_ks.h"
#include "framework/config.h"
#include "framework/environment.h"

namespace Aperture {

template <typename Conf>
grid_ks_t<Conf>::grid_ks_t(sim_environment& env, const domain_comm<Conf>* comm)  :
    grid_t<Conf>(env, comm) {
  env.params().get_value("bh_spin", a);
}

template class grid_ks_t<Config<2>>;

}
