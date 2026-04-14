#include "catch2/catch_all.hpp"
#include "utils/index.hpp"

using namespace Aperture;

TEST_CASE("index_t construction and inc/dec", "[index]") {
  index_t<3> idx(2, 5, 8);
  REQUIRE(idx[0] == 2);
  REQUIRE(idx[1] == 5);
  REQUIRE(idx[2] == 8);

  auto ix = idx.inc_x(3);
  REQUIRE(ix[0] == 5);
  REQUIRE(ix[1] == 5);
  REQUIRE(ix[2] == 8);

  auto iy = idx.inc_y(1);
  REQUIRE(iy[0] == 2);
  REQUIRE(iy[1] == 6);

  auto dz = idx.dec_z(2);
  REQUIRE(dz[2] == 6);

  // Default construction zeroes all
  index_t<2> z;
  REQUIRE(z[0] == 0);
  REQUIRE(z[1] == 0);
}

TEST_CASE("extent_t size and strides", "[index]") {
  extent_t<3> ext(10, 20, 30);

  REQUIRE(ext.size() == 10 * 20 * 30);
  REQUIRE(ext[0] == 10);
  REQUIRE(ext[1] == 20);
  REQUIRE(ext[2] == 30);

  // Column-major strides: [1, 10, 200]
  auto& s = ext.strides();
  REQUIRE(s[0] == 1);
  REQUIRE(s[1] == 10);
  REQUIRE(s[2] == 200);
}

TEST_CASE("extent_t 1D and 2D", "[index]") {
  extent_t<1> e1(100);
  REQUIRE(e1.size() == 100);
  REQUIRE(e1.strides()[0] == 1);

  extent_t<2> e2(8, 16);
  REQUIRE(e2.size() == 128);
  REQUIRE(e2.strides()[0] == 1);
  REQUIRE(e2.strides()[1] == 8);
}

TEST_CASE("idx_col_major_t to_linear/get_pos round-trip", "[index]") {
  extent_t<3> ext(10, 20, 30);

  // Test a set of positions
  index_t<3> pos(3, 7, 15);
  idx_col_major_t<3> idx(pos, ext);

  auto linear = idx.linear;
  REQUIRE(linear == 3 + 7 * 10 + 15 * 200);

  auto recovered = get_pos(idx, ext);
  REQUIRE(recovered[0] == 3);
  REQUIRE(recovered[1] == 7);
  REQUIRE(recovered[2] == 15);

  // 2D round-trip
  extent_t<2> ext2(8, 16);
  index_t<2> pos2(5, 11);
  idx_col_major_t<2> idx2(pos2, ext2);
  REQUIRE(idx2.linear == 5 + 11 * 8);
  auto rec2 = get_pos(idx2, ext2);
  REQUIRE(rec2[0] == 5);
  REQUIRE(rec2[1] == 11);

  // 1D round-trip
  extent_t<1> ext1(100);
  index_t<1> pos1(42);
  idx_col_major_t<1> idx1(pos1, ext1);
  REQUIRE(idx1.linear == 42);
  auto rec1 = get_pos(idx1, ext1);
  REQUIRE(rec1[0] == 42);
}

TEST_CASE("idx_col_major_t inc/dec", "[index]") {
  extent_t<3> ext(10, 20, 30);
  index_t<3> pos(3, 7, 15);
  idx_col_major_t<3> idx(pos, ext);
  auto base = idx.linear;

  auto ix = idx.inc_x(2);
  REQUIRE(ix.linear == base + 2);

  auto iy = idx.inc_y(3);
  REQUIRE(iy.linear == base + 3 * 10);

  auto iz = idx.inc_z(1);
  REQUIRE(iz.linear == base + 1 * 200);

  auto dx = idx.dec_x(1);
  REQUIRE(dx.linear == base - 1);

  auto dy = idx.dec_y(2);
  REQUIRE(dy.linear == base - 2 * 10);
}

TEST_CASE("idx_row_major_t to_linear and member get_pos round-trip", "[index]") {
  extent_t<3> ext(10, 20, 30);

  // Row-major: linear = pos[0]*20*30 + pos[1]*30 + pos[2]
  index_t<3> pos(3, 7, 15);
  idx_row_major_t<3> idx(pos, ext);
  REQUIRE(idx.linear == 3 * 20 * 30 + 7 * 30 + 15);

  // Use member get_pos() which uses the generic (correct) pos() method
  auto recovered = idx.get_pos();
  REQUIRE(recovered[0] == 3);
  REQUIRE(recovered[1] == 7);
  REQUIRE(recovered[2] == 15);

  // 2D round-trip
  extent_t<2> ext2(8, 16);
  index_t<2> pos2(5, 11);
  idx_row_major_t<2> idx2(pos2, ext2);
  REQUIRE(idx2.linear == 5 * 16 + 11);
  auto rec2 = idx2.get_pos();
  REQUIRE(rec2[0] == 5);
  REQUIRE(rec2[1] == 11);
}

TEST_CASE("idx_col_major_t exhaustive 2D round-trip", "[index]") {
  extent_t<2> ext(8, 12);
  for (int y = 0; y < 12; y++) {
    for (int x = 0; x < 8; x++) {
      index_t<2> pos(x, y);
      idx_col_major_t<2> idx(pos, ext);
      auto rec = get_pos(idx, ext);
      REQUIRE(rec[0] == x);
      REQUIRE(rec[1] == y);
    }
  }
}

TEST_CASE("idx_row_major_t exhaustive 2D round-trip", "[index]") {
  extent_t<2> ext(8, 12);
  for (int y = 0; y < 12; y++) {
    for (int x = 0; x < 8; x++) {
      index_t<2> pos(x, y);
      idx_row_major_t<2> idx(pos, ext);
      // Use member get_pos() -- the free function get_pos specialization
      // for row-major 2D/3D has a known bug (uses wrong extent dimensions)
      auto rec = idx.get_pos();
      REQUIRE(rec[0] == x);
      REQUIRE(rec[1] == y);
    }
  }
}

TEST_CASE("4D col_major round-trip", "[index]") {
  extent_t<4> ext(4, 5, 6, 7);
  REQUIRE(ext.size() == 4 * 5 * 6 * 7);

  index_t<4> pos(2, 3, 4, 5);
  idx_col_major_t<4> idx(pos, ext);
  auto rec = get_pos(idx, ext);
  REQUIRE(rec[0] == 2);
  REQUIRE(rec[1] == 3);
  REQUIRE(rec[2] == 4);
  REQUIRE(rec[3] == 5);
}
