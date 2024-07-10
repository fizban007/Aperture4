#include "framework/config.h"
#include "systems/policies/coord_policy_cartesian.hpp"
#include "systems/policies/exec_policy_host.hpp"
#include "systems/vlasov_solver_impl.hpp"

namespace Aperture {

template class vlasov_solver<Config<1>, 1, exec_policy_host, coord_policy_cartesian>;

}
