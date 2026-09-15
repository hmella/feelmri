#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

// Finiteness test that survives -ffinite-math-only (implied by the default
// -Ofast build), under which `v != v`, std::isnan and Eigen::allFinite() are
// all folded to false. Reads the IEEE exponent field from the object
// representation: all ones means Inf or NaN. memcpy is the standard-blessed
// spelling and compiles to a register move.
//
// Any finiteness check anywhere in cpp/ must go through this, or it is dead
// code, and that is what happened to the first b1_map guard.
template <typename T>
inline bool feelmri_is_finite(T value) {
  static_assert(std::numeric_limits<T>::is_iec559,
                "feelmri_is_finite assumes IEEE 754 binary32/binary64");
  using Bits = typename std::conditional<sizeof(T) == 4,
                                         std::uint32_t, std::uint64_t>::type;
  static_assert(sizeof(Bits) == sizeof(T), "unexpected floating-point width");
  Bits bits;
  std::memcpy(&bits, &value, sizeof(T));
  const Bits exponent = (sizeof(T) == 4)
      ? Bits(0x7F800000u) : Bits(0x7FF0000000000000ull);
  return (bits & exponent) != exponent;
}

// True for NaN only. +/-Inf is NOT NaN: an infinite T2 is the idiomatic
// spelling of "no relaxation" and inverts to exactly zero, which is correct.
template <typename T>
inline bool feelmri_is_nan(T value) {
  static_assert(std::numeric_limits<T>::is_iec559,
                "feelmri_is_nan assumes IEEE 754 binary32/binary64");
  using Bits = typename std::conditional<sizeof(T) == 4,
                                         std::uint32_t, std::uint64_t>::type;
  Bits bits;
  std::memcpy(&bits, &value, sizeof(T));
  const Bits exponent = (sizeof(T) == 4)
      ? Bits(0x7F800000u) : Bits(0x7FF0000000000000ull);
  const Bits mantissa = (sizeof(T) == 4)
      ? Bits(0x007FFFFFu) : Bits(0x000FFFFFFFFFFFFFull);
  return (bits & exponent) == exponent && (bits & mantissa) != Bits(0);
}

// True for a value strictly greater than zero, +Inf included. Written on the
// bit pattern rather than as `v > 0` because -ffinite-math-only licenses the
// compiler to reason about comparisons as if Inf and NaN could not occur,
// which is precisely the case being tested.
template <typename T>
inline bool feelmri_is_positive(T value) {
  using Bits = typename std::conditional<sizeof(T) == 4,
                                         std::uint32_t, std::uint64_t>::type;
  Bits bits;
  std::memcpy(&bits, &value, sizeof(T));
  const Bits sign = (sizeof(T) == 4)
      ? Bits(0x80000000u) : Bits(0x8000000000000000ull);
  if (feelmri_is_nan(value)) return false;
  if ((bits & sign) != Bits(0)) return false;      // negative, or -0.0
  return (bits & ~sign) != Bits(0);                // +0.0 is not positive
}
