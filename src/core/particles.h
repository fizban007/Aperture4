#ifndef __PARTICLES_H_
#define __PARTICLES_H_

#include "utils/buffer.h"
#include "particle_structs.h"

namespace Aperture {

template <typename BufferType>
class particles_base : public BufferType {
 private:
  size_t m_size = 0;
  size_t m_number = 0;

  buffer_t<size_t, BufferType::model()> m_index;
  buffer_t<double, BufferType::model()> m_tmp_data;

 public:
  typedef BufferType base_type;
  typedef particles_base<BufferType> self_type;

  particles_base() {}
  particles_base(size_t size) { resize(size); }
  particles_base(const self_type& other) = delete;
  particles_base(self_type&& other) = default;
  ~particles_base() {}

  self_type& operator=(const self_type& other) = delete;
  self_type& operator=(self_type&& other) = default;

  void resize(size_t size) {
    visit_struct::for_each(*dynamic_cast<base_type*>(this),
                           [size](const char* name, auto& x) {
                             x.resize(size);
                           });
    m_size = size;
  }

  void copy_from(const self_type& other, size_t num,
                 size_t src_pos, size_t dst_pos) {
    visit_struct::for_each(
        *dynamic_cast<base_type*>(this),
        *dynamic_cast<const base_type*>(&other),
        [num, src_pos, dst_pos](const char* name, auto& u, auto& v) {
          u.copy_from(v, num, src_pos, dst_pos);
        });
  }

  void erase(size_t pos, size_t amount = 1) {
    this->cell.assign(pos, pos + amount, empty_cell);
  }

  void init() {
    erase(0, m_size);
  }

  void copy_to_host() {
    visit_struct::for_each(*dynamic_cast<base_type*>(this),
                           [](const char* name, auto& x) {
                             x.copy_to_host();
                           });
  }

  void copy_to_device() {
    visit_struct::for_each(*dynamic_cast<base_type*>(this),
                           [](const char* name, auto& x) {
                             x.copy_to_device();
                           });
  }

  size_t size() { return m_size; }
  size_t number() { return m_number; }

  void set_num(size_t num) {
    // Can't set a number larger than maximum size
    m_number = std::min(num, m_size);
  }
};


#ifdef CUDA_ENABLED

template <MemoryModel Model = MemoryModel::host_device>
using particles_t = particles_base<ptc_buffer<Model>>;

template <MemoryModel Model = MemoryModel::host_device>
using photons_t = particles_base<ph_buffer<Model>>;

#else

template <MemoryModel Model = MemoryModel::host_only>
using particles_t = particles_base<ptc_buffer<Model>>;

template <MemoryModel Model = MemoryModel::host_only>
using photons_t = particles_base<ph_buffer<Model>>;

#endif

}

#endif
