#include <cassert>
#include <string>
#include <utility>
#include <type_traits>
#include <iostream>

#include "column.h"

// ---------- tiny helpers ----------
template <class T, class U>
static void expect_eq(const T& a, const U& b, const char* msg) {
  if (!(a == b)) {
    std::cerr << "EXPECT_EQ failed: " << msg << "\n";
    std::abort();
  }
}

// ============= TESTS =============

// 1) basic: reserve/size/push_back/operator[]
static void test_basic_int() {
  SRDatalog::Column<int> c;
  assert(c.size() == 0);

  c.reserve(10);
  // reserve should not change size
  assert(c.size() == 0);

  c.push_back(1);
  c.push_back(2);
  c.push_back(3);

  assert(c.size() == 3);
  expect_eq(c[0], 1, "c[0]");
  expect_eq(c[1], 2, "c[1]");
  expect_eq(c[2], 3, "c[2]");

  // random access should be mutable
  c[1] = 42;
  expect_eq(c[1], 42, "mutate via operator[]");
}

// 2) push_back rvalues and lvalues; data() consistency
static void test_push_value_categories() {
  SRDatalog::Column<std::string> s;
  std::string a = "hello";
  const std::string b = "world";

  // lvalue
  s.push_back(a);
  // const lvalue
  s.push_back(b);
  // rvalue
  s.push_back(std::string("!"));

  assert(s.size() == 3);
  expect_eq(s[0], std::string("hello"), "s[0]");
  expect_eq(s[1], std::string("world"), "s[1]");
  expect_eq(s[2], std::string("!"),     "s[2]");

  // data() points to first element; matches operator[]
  assert(s.data() != nullptr);
  expect_eq(*(s.data()+0), s[0], "data()[0] equals operator[]");
  expect_eq(*(s.data()+1), s[1], "data()[1] equals operator[]");
  expect_eq(*(s.data()+2), s[2], "data()[2] equals operator[]");
}

// 3) iteration works (non-const and const)
static void test_iteration() {
  SRDatalog::Column<int> c;
  for (int i = 0; i < 5; ++i) c.push_back(i);

  int sum = 0;
  for (int x : c) sum += x;
  expect_eq(sum, 0+1+2+3+4, "range-for sum");

  const auto& cc = c;
  int prod = 1;
  for (int x : cc) prod *= (x+1); // 1*2*3*4*5 = 120
  expect_eq(prod, 120, "const range-for product");
}

// 4) large reserve then append, to exercise growth policy
static void test_reserve_then_fill() {
  SRDatalog::Column<int> c;
  c.reserve(1000);
  for (int i = 0; i < 1000; ++i) c.push_back(i);
  assert(c.size() == 1000);
  expect_eq(c[0],   0,   "first");
  expect_eq(c[999], 999, "last");
}

// 5) sanity on trivial types vs. std::string + emplace_back
struct Point { int x; int y; };
static void test_structs_and_strings() {
  static_assert(std::is_trivially_copyable_v<Point>, "Point should be trivially copyable");

  SRDatalog::Column<Point> pts;
  pts.emplace_back(1,2);
  pts.emplace_back(3,4);
  assert(pts.size() == 2);
  assert(pts[0].x == 1 && pts[0].y == 2);
  assert(pts[1].x == 3 && pts[1].y == 4);

  SRDatalog::Column<std::string> s;
  s.emplace_back("a");
  s.emplace_back("b");
  assert(s.size() == 2);
  expect_eq(s[0], std::string("a"), "string a");
  expect_eq(s[1], std::string("b"), "string b");
}

// 6) clear() behavior
static void test_clear() {
  SRDatalog::Column<int> c;
  c.push_back(10);
  c.push_back(20);
  c.push_back(30);
  assert(c.size() == 3);

  c.clear();
  assert(c.size() == 0);

  // After clear, pushing still works
  c.push_back(99);
  assert(c.size() == 1);
  expect_eq(c[0], 99, "after clear -> push_back");
}

// 7) address stability under push_back (reallocation allowed)
static void test_address_stability_under_push() {
  SRDatalog::Column<int> c;
  c.reserve(2);
  c.push_back(10);
  c.push_back(20);
  int* p0 = c.data();
  // May reallocate here; behavior is allowed. We just ensure values survive.
  c.push_back(30);
  assert(c.size() == 3);
  expect_eq(c[0], 10, "value survives");
  expect_eq(c[1], 20, "value survives");
  expect_eq(c[2], 30, "value survives");
  (void)p0; // we don't assert pointer equality because reallocation is OK
}

// 8) stress: many small push_backs
static void test_many_small_pushes() {
  SRDatalog::Column<int> c;
  for (int i = 0; i < 10000; ++i) c.push_back(i);
  assert(c.size() == 10000);
  expect_eq(c[1234], 1234, "random index");
  expect_eq(c[9999], 9999, "last index");
}

int main() {
  test_basic_int();
  test_push_value_categories();
  test_iteration();
  test_reserve_then_fill();
  test_structs_and_strings();
  test_clear();
  test_address_stability_under_push();
  test_many_small_pushes();
  std::cout << "[OK] All Column<T> tests passed.\n";
  return 0;
}
