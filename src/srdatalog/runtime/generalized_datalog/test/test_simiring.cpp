// test_semiring.cpp
#include <cassert>
#include <iostream>
#include <type_traits>

#include "semiring.h"

// ---------- tiny helpers ----------
template <class T>
static void expect_eq(const T& a, const T& b, const char* msg) {
  if (!(a == b)) {
    std::cerr << "EXPECT_EQ failed: " << msg << " | left=" << a << " right=" << b << "\n";
    std::abort();
  }
}

template <class T>
static void expect_true(bool cond, const char* msg, const T& a, const T& b) {
  if (!cond) {
    std::cerr << "EXPECT_TRUE failed: " << msg << " | a=" << a << " b=" << b << "\n";
    std::abort();
  }
}

// =======================================================
// 1) Concept conformance (compile-time)
// =======================================================
static void test_concept_conformance() {
  static_assert(Semiring<NaturalBag>, "NaturalBag must satisfy Semiring");
  static_assert(Semiring<Bag>, "Bag alias must satisfy Semiring");
  static_assert(Semiring<BooleanSR>, "BooleanSR must satisfy Semiring");
  static_assert(Semiring<ProbIndep>, "ProbIndep must satisfy Semiring");
  static_assert(Semiring<NoProvenance>, "NoProvenance must satisfy Semiring");

  // NoProvenance type properties
  static_assert(sizeof(semiring_value_t<NoProvenance>) == 1,
                "NoProvenance::value_type (monostate) should be 1 byte");

  // has_provenance_v trait
  static_assert(has_provenance_v<BooleanSR>, "BooleanSR has provenance");
  static_assert(has_provenance_v<NaturalBag>, "NaturalBag has provenance");
  static_assert(has_provenance_v<ProbIndep>, "ProbIndep has provenance");
  static_assert(!has_provenance_v<NoProvenance>, "NoProvenance must NOT have provenance");
}

// =======================================================
// 2) NaturalBag / Bag semantics (ℕ, +, *)
// =======================================================
static void test_naturalbag_basic() {
  using SR = NaturalBag;
  using T = SR::value_type;

  // identities
  expect_eq(SR::zero(), T(0), "Bag::zero");
  expect_eq(SR::one(), T(1), "Bag::one");

  // add/mul
  expect_eq(SR::add(2, 3), T(5), "Bag add");
  expect_eq(SR::mul(2, 3), T(6), "Bag mul");

  // wrappers
  expect_eq(sr_zero<SR>(), T(0), "sr_zero");
  expect_eq(sr_one<SR>(), T(1), "sr_one");
  expect_eq(sr_add<SR>(4, 5), T(9), "sr_add");
  expect_eq(sr_mul<SR>(4, 5), T(20), "sr_mul");

  // simple properties (commutativity of mul per default trait)
  expect_eq(SR::mul(7, 11), SR::mul(11, 7), "Bag mul commutative");
}

// =======================================================
// 3) Boolean semiring (∨, ∧) and idempotence of add
// =======================================================
static void test_boolean_basic() {
  using SR = BooleanSR;
  using T = SR::value_type;

  expect_eq(SR::zero(), false, "Bool::zero == false");
  expect_eq(SR::one(), true, "Bool::one  == true");

  // add is OR, mul is AND
  expect_eq(SR::add(false, false), false, "OR ff");
  expect_eq(SR::add(true, false), true, "OR tf");
  expect_eq(SR::add(true, true), true, "OR tt");

  expect_eq(SR::mul(true, true), true, "AND tt");
  expect_eq(SR::mul(true, false), false, "AND tf");
  expect_eq(SR::mul(false, false), false, "AND ff");

  // idempotence of add: a ⊕ a = a
  expect_eq(SR::add(false, false), false, "idempotent add (false)");
  expect_eq(SR::add(true, true), true, "idempotent add (true)");

  // trait flags
  static_assert(semiring_traits<SR>::add_idempotent, "Boolean add should be idempotent");
  static_assert(semiring_traits<SR>::mul_commutative, "Boolean mul should be commutative");
}

// =======================================================
// 4) ProbIndep semiring: a⊕b = 1 - (1-a)(1-b), a⊗b = ab with clamping
// =======================================================
static void test_probindep_basic() {
  using SR = ProbIndep;
  using T = SR::value_type;

  expect_eq(SR::zero(), T(0.0), "Prob::zero");
  expect_eq(SR::one(), T(1.0), "Prob::one");

  // add: 0.9 ⊕ 0.9 = 0.99
  {
    T a = 0.9, b = 0.9;
    T r = SR::add(a, b);
    expect_true(std::abs(r - 0.99) < 1e-12, "Prob add 0.9⊕0.9 ≈ 0.99", r, T(0.99));
  }

  // mul: 0.8 ⊗ 0.8 = 0.64
  {
    T a = 0.8, b = 0.8;
    T r = SR::mul(a, b);
    expect_true(std::abs(r - 0.64) < 1e-12, "Prob mul 0.8⊗0.8 ≈ 0.64", r, T(0.64));
  }

  // clamping: values outside [0,1] get clamped back
  {
    T r1 = SR::mul(1.2, 0.9);    // 1.08 -> clamp to 1.0
    T r2 = SR::add(-0.1, -0.3);  // -0.1 + -0.3 - product = -0.37 -> clamp to 0.0
    expect_eq(r1, T(1.0), "Prob mul clamp high");
    expect_eq(r2, T(0.0), "Prob add clamp low");
  }

  // wrappers
  expect_eq(sr_add<SR>(0.25, 0.5), T(0.625), "sr_add Prob");
  expect_eq(sr_mul<SR>(0.25, 0.5), T(0.125), "sr_mul Prob");

  // trait flags
  static_assert(!semiring_traits<SR>::add_idempotent, "Prob add is not idempotent");
  static_assert(semiring_traits<SR>::mul_commutative, "Prob mul should be commutative");
}

// =======================================================
// 5) NoProvenance semiring (sentinel type with zero-cost operations)
// =======================================================
static void test_noprovenance_basic() {
  using SR = NoProvenance;
  using T = SR::value_type;

  // value_type is std::monostate - verify zero-cost storage
  static_assert(std::is_same_v<T, std::monostate>, "NoProvenance::value_type must be monostate");
  static_assert(sizeof(T) == 1, "monostate should be 1 byte");
  static_assert(std::is_empty_v<T>, "monostate should be empty");

  // All operations return empty monostate - they compile and are no-ops
  T zero_val = SR::zero();
  T one_val = SR::one();
  T add_result = SR::add(zero_val, one_val);
  T mul_result = SR::mul(zero_val, one_val);

  // All results are the same empty monostate (use assert since monostate lacks operator<<)
  assert(zero_val == one_val && "NoProvenance zero == one (all monostate)");
  assert(add_result == mul_result && "NoProvenance add == mul (all monostate)");

  // sr_* wrappers also work
  assert(sr_zero<SR>() == T{} && "sr_zero<NoProvenance>");
  assert(sr_one<SR>() == T{} && "sr_one<NoProvenance>");
  assert(sr_add<SR>(T{}, T{}) == T{} && "sr_add<NoProvenance>");
  assert(sr_mul<SR>(T{}, T{}) == T{} && "sr_mul<NoProvenance>");

  // has_provenance_v must be false
  static_assert(!has_provenance_v<SR>, "NoProvenance has_provenance_v must be false");

  // Other semirings have provenance
  static_assert(has_provenance_v<BooleanSR>, "BooleanSR has provenance");
  static_assert(has_provenance_v<NaturalBag>, "NaturalBag has provenance");

  std::cout << "  NoProvenance tests passed.\n";
}

// =======================================================
// 5) Generic sanity via templates (works for all three SRs)
// =======================================================
template <Semiring SR>
static void generic_laws_smoke() {
  using T = typename SR::value_type;

  // neutral elements
  {
    T a = SR::one();
    T z = SR::zero();
    expect_eq(SR::add(a, z), a, "a ⊕ 0 = a");
    expect_eq(SR::mul(a, SR::one()), a, "a ⊗ 1 = a");
  }

  // simple commutativity checks where trait says mul is commutative
  if constexpr (semiring_traits<SR>::mul_commutative) {
    T x = SR::one();
    T y = SR::one();
    expect_eq(SR::mul(x, y), SR::mul(y, x), "⊗ commutative (smoke)");
  }
}

int main() {
  test_concept_conformance();

  test_naturalbag_basic();
  test_boolean_basic();
  test_probindep_basic();
  test_noprovenance_basic();

  // generic smoke across all
  generic_laws_smoke<NaturalBag>();
  generic_laws_smoke<BooleanSR>();
  generic_laws_smoke<ProbIndep>();

  std::cout << "[OK] semiring.h tests passed.\n";
  return 0;
}
