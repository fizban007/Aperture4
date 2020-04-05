#include "particles.h"
#include "visit_struct/visit_struct.hpp"

namespace Aperture {

template <typename BufferType>
particles_base<BufferType>::particles_base() :
    m_size(0), m_number(0) {}

template <typename BufferType>
particles_base<BufferType>::particles_base(size_t size) {
  resize(size);
}

template <typename BufferType>
particles_base<BufferType>::~particles_base() {}

template <typename BufferType>
void
particles_base<BufferType>::resize(size_t size) {
  visit_struct::for_each(*dynamic_cast<base_type*>(this),
                         [size](const char* name, auto& x) {
                           x.resize(size);
                         });
  m_size = size;
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_host() {
  visit_struct::for_each(*dynamic_cast<base_type*>(this),
                         [](const char* name, auto& x) {
                           x.copy_to_host();
                         });
}

template <typename BufferType>
void
particles_base<BufferType>::copy_to_device() {
  visit_struct::for_each(*dynamic_cast<base_type*>(this),
                         [](const char* name, auto& x) {
                           x.copy_to_device();
                         });
}

// Explicit instantiation
template class particles_base<ptc_buffer<MemoryModel::host_only>>;
template class particles_base<ptc_buffer<MemoryModel::host_device>>;
template class particles_base<ptc_buffer<MemoryModel::device_managed>>;
template class particles_base<ptc_buffer<MemoryModel::device_only>>;

template class particles_base<ph_buffer<MemoryModel::host_only>>;
template class particles_base<ph_buffer<MemoryModel::host_device>>;
template class particles_base<ph_buffer<MemoryModel::device_managed>>;
template class particles_base<ph_buffer<MemoryModel::device_only>>;

}
