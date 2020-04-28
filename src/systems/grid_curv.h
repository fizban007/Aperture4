#ifndef _GRID_CURV_H_
#define _GRID_CURV_H_

#include "core/multi_array.hpp"
#include "grid.h"

namespace Aperture {

template <typename ValueT, int Rank, typename Idx_t>
struct grid_ptrs {
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> le;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> lb;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> Ae;
  vec_t<ndptr_const<ValueT, Rank, Idx_t>, 3> Ab;
  ndptr_const<ValueT, Rank, Idx_t> dV;
};

template <typename Conf>
class grid_curv_t : public grid_t<Conf> {
 public:
  static std::string name() { return "grid"; }
  typedef typename Conf::value_t value_t;
  typedef grid_ptrs<value_t, Conf::dim, typename Conf::idx_t>
      grid_ptrs_t;

  grid_curv_t(
      sim_environment &env,
      const domain_info_t<Conf::dim> &domain_info = domain_info_t<Conf::dim>{}) :
      grid_t<Conf>(env, domain_info) {}

  grid_curv_t(sim_environment &env, const domain_comm<Conf> &comm) :
      grid_t<Conf>(env, comm) {}
  virtual ~grid_curv_t() {}

  void init() {
    for (int i = 0; i < 3; i++) {
      m_le[i].resize(this->extent());
      m_lb[i].resize(this->extent());
      m_Ae[i].resize(this->extent());
      m_Ab[i].resize(this->extent());
    }
    m_dV.resize(this->extent());

    compute_coef();
  }

  virtual void compute_coef() = 0;
  grid_ptrs_t get_grid_ptrs() const {
    grid_ptrs_t result;

    for (int i = 0; i < 3; i++) {
      result.le[i] = m_le[i].get_const_ptr();
      result.lb[i] = m_lb[i].get_const_ptr();
      result.Ae[i] = m_Ae[i].get_const_ptr();
      result.Ab[i] = m_Ab[i].get_const_ptr();
    }
    result.dV = m_dV.get_const_ptr();

    return result;
  }

  std::array<multi_array<value_t, Conf::dim>, 3> m_le;
  std::array<multi_array<value_t, Conf::dim>, 3> m_lb;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ae;
  std::array<multi_array<value_t, Conf::dim>, 3> m_Ab;
  multi_array<value_t, Conf::dim> m_dV;
};


}

#endif  // _GRID_CURV_H_
