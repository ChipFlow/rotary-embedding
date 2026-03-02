#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_BFLOAT__)

typedef bfloat bfloat16_t;

#else

constexpr METAL_FUNC uint16_t float_to_bfloat_bits(float x) {
  if ((as_type<uint32_t>(x) & ~_fp_encoding_traits<float>::sign_mask) >
      _fp_encoding_traits<float>::inf_mask) {
    return uint16_t(as_type<uint32_t>(0x7FC0));
  }
  uint32_t float_bits = as_type<uint32_t>(x);
  float_bits += ((float_bits >> 16) & 1) + as_type<uint32_t>(0x7FFF);
  return float_bits >> 16;
}

constexpr METAL_FUNC float bfloat_bits_to_float(uint16_t x) {
  return as_type<float>((uint32_t)x << 16);
}

struct _MLX_BFloat16;

template <typename T>
static constexpr constant bool can_convert_to_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_bfloat =
    !is_same_v<T, _MLX_BFloat16> && is_convertible_v<float, T>;

struct _MLX_BFloat16 {
  uint16_t bits_;
  _MLX_BFloat16() thread = default;
  _MLX_BFloat16() threadgroup = default;
  _MLX_BFloat16() device = default;
  _MLX_BFloat16() constant = default;

  struct bits_to_bfloat_struct {};
  static constexpr METAL_FUNC bits_to_bfloat_struct bits_to_bfloat() {
    return bits_to_bfloat_struct();
  }
  constexpr METAL_FUNC _MLX_BFloat16(uint16_t bits, bits_to_bfloat_struct)
      : bits_(bits) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) thread
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) threadgroup
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) device
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_to_bfloat<T>>::type>
  constexpr METAL_FUNC _MLX_BFloat16(T x) constant
      : bits_(float_to_bfloat_bits(static_cast<float>(x))) {}

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const thread {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const threadgroup {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() const device {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }

  template <typename T,
            typename = typename enable_if<can_convert_from_bfloat<T>>::type>
  constexpr METAL_FUNC operator T() constant {
    return static_cast<T>(bfloat_bits_to_float(bits_));
  }
};

constexpr METAL_FUNC _MLX_BFloat16 operator-(_MLX_BFloat16 x) {
  return -static_cast<float>(x);
}

#define bfloat_binop_base(__op__, __operator__, otype, atype, btype, ctype)    \
  constexpr METAL_FUNC otype __operator__(atype lhs, btype rhs) {              \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop_helper(__op__, __operator__, otype, itype, ctype)         \
  constexpr METAL_FUNC otype __operator__(_MLX_BFloat16 lhs, itype rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }                                                                            \
  constexpr METAL_FUNC otype __operator__(itype lhs, _MLX_BFloat16 rhs) {      \
    return static_cast<ctype>(lhs) __op__ static_cast<ctype>(rhs);             \
  }

#define bfloat_binop(_op_, _operator_)                                         \
  bfloat_binop_base(_op_, _operator_, _MLX_BFloat16, _MLX_BFloat16,            \
                    _MLX_BFloat16, float);                                     \
  bfloat_binop_helper(_op_, _operator_, float, float, float);                  \
  bfloat_binop_helper(_op_, _operator_, float, half, float);                   \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int32_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint32_t, float);       \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, int64_t, float);        \
  bfloat_binop_helper(_op_, _operator_, _MLX_BFloat16, uint64_t, float);

bfloat_binop(+, operator+);
bfloat_binop(-, operator-);
bfloat_binop(*, operator*);
bfloat_binop(/, operator/);

#undef bfloat_binop_base
#undef bfloat_binop_helper
#undef bfloat_binop

typedef struct _MLX_BFloat16 bfloat16_t;

#endif
