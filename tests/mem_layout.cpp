#include "morton2d.h"
#include "morton3d.h"
#include "utils/index.h"
#include "utils/timer.h"
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace Aperture;

double
interp(double* f, double x, double y, double z, uint64_t idx,
       uint32_t N1, uint32_t N2) {
  int di_m = 0;
  int di_p = 1;
  int dj_m = 0;
  int dj_p = 1;
  int dk_m = 0;
  int dk_p = 1;

  double f11 = (1.0 - z) * f[idx + di_p + dj_p * N1 + dk_m * N1 * N2] +
               z * f[idx + di_p + dj_p * N1 + dk_p * N1 * N2];
  double f10 = (1.0 - z) * f[idx + di_p + dj_m * N1 + dk_m * N1 * N2] +
               z * f[idx + di_p + dj_m * N1 + dk_p * N1 * N2];
  double f01 = (1.0 - z) * f[idx + di_m + dj_p * N1 + dk_m * N1 * N2] +
               z * f[idx + di_m + dj_p * N1 + dk_p * N1 * N2];
  double f00 = (1.0 - z) * f[idx + di_m + dj_m * N1 + dk_m * N1 * N2] +
               z * f[idx + di_m + dj_m * N1 + dk_p * N1 * N2];
  double f1 = y * f11 + (1.0 - y) * f10;
  double f0 = y * f01 + (1.0 - y) * f00;
  return x * f1 + (1.0 - x) * f0;
}

template <typename Index_t>
double
interp(double* f, double x, double y, double z, const Index_t& idx) {
  int di_m = 0;
  int di_p = 1;
  int dj_m = 0;
  int dj_p = 1;
  int dk_m = 0;
  int dk_p = 1;

  double f11 = (1.0 - z) * f[idx.incX().incY().key] +
               z * f[idx.incX().incY().incZ().key];
  double f10 = (1.0 - z) * f[idx.incX().key] + z * f[idx.incX().incZ().key];
  double f01 = (1.0 - z) * f[idx.incY().key] + z * f[idx.incY().incZ().key];
  double f00 = (1.0 - z) * f[idx.key] + z * f[idx.incZ().key];
  double f1 = y * f11 + (1.0 - y) * f10;
  double f0 = y * f01 + (1.0 - y) * f00;
  return x * f1 + (1.0 - x) * f0;
}

// double
// interp_morton(double* f, double x, double y, double z, morton3& m) {
//   double f11 = (1.0 - z) * f[m.incX().incY().key] +
//                z * f[m.incX().incY().incZ().key];
//   double f10 = (1.0 - z) * f[m.incX().key] + z * f[m.incX().incZ().key];
//   double f01 = (1.0 - z) * f[m.incY().key] + z * f[m.incY().incZ().key];
//   double f00 = (1.0 - z) * f[m.key] + z * f[m.incZ().key];
//   double f1 = y * f11 + (1.0 - y) * f10;
//   double f0 = y * f01 + (1.0 - y) * f00;
//   return x * f1 + (1.0 - x) * f0;
// }

int
main(int argc, char* argv[]) {
  std::default_random_engine g;
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::uniform_int_distribution<int> dN(1, 63);

  // uint32_t N = 2048 * 2048;
  uint32_t N1 = 512;
  uint32_t N2 = 512;
  uint32_t N3 = 512;

  vector<double> v1(N1 * N2 * N3);
  vector<double> v2(N1 * N2 * N3);
  vector<double> u(N1 * N2 * N3);

  for (int k = 0; k < N3; k++) {
    for (int j = 0; j < N2; j++) {
      for (int i = 0; i < N1; i++) {
        morton3 m = morton3(i, j, k);
        v1[m.key] = i - j + k;
        v2[i + j * N1 + k * N1 * N2] = i - j + k;
      }
    }
  }

  int M = 1000;
  vector<double> xs(M);
  vector<double> ys(M);
  vector<double> zs(M);
  for (int n = 0; n < M; n++) {
    xs[n] = dist(g);
    ys[n] = dist(g);
    zs[n] = dist(g);
  }

  timer::stamp();
  for (int t = 0; t < 10; t++) {
    int k0 = dN(g) * 8, j0 = dN(g) * 8, i0 = dN(g) * 8;
    for (int k = k0; k < k0 + 8; k++) {
      for (int j = j0; j < j0 + 8; j++) {
        for (int i = i0; i < i0 + 8; i++) {
          // Do interpolation M times
          // uint64_t idx = i + j * N1 + k * N1 * N2;
          idx_col_major_t<> idx(i, j, k, Extent{N1, N2, 1u});
          for (int n = 0; n < M; n++) {
            u[idx.key] =
                interp(v2.data(), xs[n], ys[n], zs[n], idx);
          }
        }
      }
    }
  }
  timer::show_duration_since_stamp("normal indexing", "ms");

  timer::stamp();
  for (int t = 0; t < 10; t++) {
    int k0 = dN(g) * 8, j0 = dN(g) * 8, i0 = dN(g) * 8;
    for (int k = k0; k < k0 + 8; k++) {
      for (int j = j0; j < j0 + 8; j++) {
        for (int i = i0; i < i0 + 8; i++) {
          morton3 m = morton3(i, j, k);
          for (int n = 0; n < M; n++) {
            u[m.key] = interp(v1.data(), xs[n], ys[n], zs[n], m);
          }
        }
      }
    }
  }
  timer::show_duration_since_stamp("morton indexing", "ms");
  return 0;
}
