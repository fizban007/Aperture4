#pragma once

#include "systems/vlasov_solver.h"

namespace Aperture {

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::vlasov_solver(const grid_t<Conf>& grid,
                                                                   const domain_comm<Conf, ExecPolicy>* comm) :
                                                                   m_grid(grid), m_comm(comm) {

                                                                   }

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::~vlasov_solver() {}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
void
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::init() {

}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
void
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::update(double dt, uint32_t step) {}

template <typename Conf,
          int Dim_P,
          template <class> class ExecPolicy,
          template <class> class CoordPolicy>
void
vlasov_solver<Conf, Dim_P, ExecPolicy, CoordPolicy>::register_data_components() {

}


}
