#pragma once

#include "core/gpu_translation_layer.h"
#include "core/data_adapter.h"
#include "core/enum_types.h"
#include "core/exec_tags.h"
#include "utils/range.hpp"

// Lightweight execution policies for the prismatic mesh code.
// These mirror exec_policy_host/gpu but without the Conf/Grid dependency.

namespace Aperture {

// =========================================================================
// Host execution policy
// =========================================================================
struct prismatic_exec_policy_host {
  using exec_tag = exec_tags::host;

  template <typename Func, typename... Args>
  static void launch(const Func& f, Args&&... args) {
    f(adapt(exec_tags::host{}, args)...);
  }

  template <typename Func, typename Idx>
  static void loop(Idx begin, Idx end, const Func& f) {
    for (auto idx : range(begin, end)) {
      f(idx);
    }
  }

  static void sync() {}

  static MemType data_mem_type() { return MemType::host_only; }
};

// =========================================================================
// GPU execution policy (only available when CUDA/HIP is enabled)
// =========================================================================
#ifdef GPU_ENABLED

struct prismatic_exec_policy_gpu {
  using exec_tag = exec_tags::device;

  template <typename Func, typename... Args>
  static void launch(const Func& f, Args&&... args) {
    kernel_launch(f, adapt(exec_tags::device{}, args)...);
    GpuCheckError();
  }

  template <typename Func, typename Idx>
  static HD_INLINE void loop(Idx begin, Idx end, const Func& f) {
    for (auto idx : grid_stride_range(begin, end)) {
      f(idx);
    }
  }

  static void sync() {
    GpuSafeCall(gpuDeviceSynchronize());
    GpuCheckError();
  }

  static MemType data_mem_type() { return MemType::host_device; }
};

using prismatic_exec_policy_dynamic = prismatic_exec_policy_gpu;

#else

using prismatic_exec_policy_dynamic = prismatic_exec_policy_host;

#endif

}  // namespace Aperture
