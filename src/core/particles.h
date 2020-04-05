#ifndef __PARTICLES_H_
#define __PARTICLES_H_

#include "buffer.h"
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

  particles_base();
  particles_base(size_t size);
  particles_base(const self_type& other) = delete;
  particles_base(self_type&& other) = default;
  ~particles_base();

  self_type& operator=(const self_type& other) = delete;
  self_type& operator=(self_type&& other) = default;

  void resize(size_t size);
  void init();
  void copy_from(const self_type& other, size_t num,
                 size_t src_pos, size_t dst_pos);
  void erase(size_t pos, size_t amount = 1);

  void copy_to_host();
  void copy_to_device();

  size_t size() { return m_size; }
  size_t number() { return m_number; }

  void set_num(size_t num);
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
