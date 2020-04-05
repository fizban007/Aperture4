#ifndef _ENUM_TYPES_H_
#define _ENUM_TYPES_H_

#include <cstdint>
#include <string>

namespace Aperture {

/// Field staggering type
enum class FieldType : char { face_centered = 0, edge_centered = 1 };

/// Particle types
enum class PtcType : unsigned char { electron = 0, positron, ion };

inline std::string ptc_type_name(int type) {
  if (type == (int)PtcType::electron) {
    return "electron";
  } else if (type == (int)PtcType::positron) {
    return "positron";
  } else if (type == (int)PtcType::ion) {
    return "ion";
  } else if (type == (int)PtcType::ion + 1) {
    return "photon";
  } else {
    return "unknown";
  }
}

enum class CommTags : char { left = 0, right };

enum class Zone : char { center = 13 };

enum class BoundaryPos : char {
  lower0,
  upper0,
  lower1,
  upper1,
  lower2,
  upper2
};

// Use util functions check_bit, set_bit, bit_or, clear_bit, and
// toggle_bit to interact with particle and photon flags. These are
// defined from lower bits.
enum class PtcFlag : uint32_t {
  nothing = 0,
  tracked = 1,
  ignore_force,
  ignore_current,
  ignore_EM,
  ignore_radiation,
  primary,
  secondary,
  annihilate,
  emit_photon
};

enum class PhFlag : uint32_t { tracked = 1, ignore_pair_create };

}  // namespace Aperture

#endif  // _ENUM_TYPES_H_
