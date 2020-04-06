#include "core/particles.h"
#include "morton2d.h"
#include "ndarray.hpp"
#include "utils/logger.h"
#include "visit_struct/visit_struct.hpp"
#include <any>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Aperture;
using namespace nd;

// The goal here is to define two systems. Each of them has some
// requirement in data components. Then build a `data` object that
// contain the required components.
//
template <template <typename> typename... Skills>
class X : public Skills<X<Skills...>>... {
 public:
  void basicMethod();
};

template <typename T, typename... Ts, typename Func>
void
variadic_loop(Func f, T& t, Ts&... ts) {
  f(t);
  variadic_loop(f, ts...);
}

template <typename T, typename Func>
void
variadic_loop(Func f, T& t) {
  f(t);
}

int
main(int argc, char* argv[]) {
  auto array =
      make_array([](auto idx) { return morton2(idx[0], idx[1]).key; },
                 make_shape(8, 8))
          .unique();
  // Logger::print_info("{}", arr(make_index(5, 5)));
  for (auto idx : array.indexes()) {
    array(idx) = morton2(idx[0], idx[1]).key;
  }

  for (int i = 0; i < 64; i++) {
    uint64_t x, y;
    morton2(i).decode(x, y);
    Logger::print_info("{}, {}: i: {}, value: {}", x, y, i,
                       array.data()[i]);
  }
  return 0;
}
