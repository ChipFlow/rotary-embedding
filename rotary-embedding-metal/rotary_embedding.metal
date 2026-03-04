#include <metal_stdlib>
#include "utils.metal"

using namespace metal;

// Function constants for compile-time specialization.
// IS_NEOX: true for GPT-NeoX style (Llama, Mistral), false for GPT-J style.
constant bool IS_NEOX [[function_constant(0)]];

// Rotary embedding kernel.
//
// Each threadgroup processes one token. Threads within the threadgroup
// are mapped to (head_idx, rot_offset) pairs covering both query and key.
//
// The cos_sin_cache layout is [max_position, rot_dim] where:
//   cache[pos, 0:rot_dim/2] = cos values
//   cache[pos, rot_dim/2:rot_dim] = sin values
//
// For NeoX style (IS_NEOX=true):
//   x_index = rot_offset, y_index = embed_dim + rot_offset
// For GPT-J style (IS_NEOX=false):
//   x_index = 2 * rot_offset, y_index = 2 * rot_offset + 1
template <typename scalar_t>
kernel void rotary_embedding_kernel(
    const device int64_t *positions [[buffer(0)]],
    device scalar_t *query [[buffer(1)]],
    device scalar_t *key [[buffer(2)]],
    const device scalar_t *cos_sin_cache [[buffer(3)]],
    const device int &rot_dim [[buffer(4)]],
    const device int64_t &query_stride [[buffer(5)]],
    const device int64_t &key_stride [[buffer(6)]],
    const device int &head_size [[buffer(7)]],
    const device int &num_heads [[buffer(8)]],
    const device int &num_kv_heads [[buffer(9)]],
    const device int &has_key [[buffer(10)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_tg [[threads_per_threadgroup]]) {

  const int embed_dim = rot_dim / 2;
  const int64_t pos = positions[token_idx];
  const device scalar_t *cache_ptr = cos_sin_cache + pos * rot_dim;

  // Process query heads.
  for (int i = tid; i < num_heads * embed_dim; i += threads_per_tg) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;

    int x_index, y_index;
    if (IS_NEOX) {
      x_index = rot_offset;
      y_index = embed_dim + rot_offset;
    } else {
      x_index = 2 * rot_offset;
      y_index = 2 * rot_offset + 1;
    }

    const int64_t token_head = token_idx * query_stride + head_idx * head_size;

    const float cos_val = static_cast<float>(cache_ptr[rot_offset]);
    const float sin_val = static_cast<float>(cache_ptr[embed_dim + rot_offset]);

    const float x = static_cast<float>(query[token_head + x_index]);
    const float y = static_cast<float>(query[token_head + y_index]);
    // Use FMA (fused multiply-add) for better precision
    query[token_head + x_index] = static_cast<scalar_t>(fma(x, cos_val, -y * sin_val));
    query[token_head + y_index] = static_cast<scalar_t>(fma(y, cos_val, x * sin_val));
  }

  // Process key heads (if key is provided).
  if (has_key) {
    for (int i = tid; i < num_kv_heads * embed_dim; i += threads_per_tg) {
      const int head_idx = i / embed_dim;
      const int rot_offset = i % embed_dim;

      int x_index, y_index;
      if (IS_NEOX) {
        x_index = rot_offset;
        y_index = embed_dim + rot_offset;
      } else {
        x_index = 2 * rot_offset;
        y_index = 2 * rot_offset + 1;
      }

      const int64_t token_head = token_idx * key_stride + head_idx * head_size;

      const float cos_val = static_cast<float>(cache_ptr[rot_offset]);
      const float sin_val = static_cast<float>(cache_ptr[embed_dim + rot_offset]);

      const float x = static_cast<float>(key[token_head + x_index]);
      const float y = static_cast<float>(key[token_head + y_index]);
      // Use FMA (fused multiply-add) for better precision
      key[token_head + x_index] = static_cast<scalar_t>(fma(x, cos_val, -y * sin_val));
      key[token_head + y_index] = static_cast<scalar_t>(fma(y, cos_val, x * sin_val));
    }
  }
}

// Instantiate kernel variants for each dtype.
#define instantiate_rotary_embedding(type)                                     \
  template [[host_name("rotary_embedding_" #type)]] [[kernel]] void            \
  rotary_embedding_kernel<type>(                                               \
      const device int64_t *positions [[buffer(0)]],                           \
      device type *query [[buffer(1)]],                                        \
      device type *key [[buffer(2)]],                                          \
      const device type *cos_sin_cache [[buffer(3)]],                          \
      const device int &rot_dim [[buffer(4)]],                                 \
      const device int64_t &query_stride [[buffer(5)]],                        \
      const device int64_t &key_stride [[buffer(6)]],                          \
      const device int &head_size [[buffer(7)]],                               \
      const device int &num_heads [[buffer(8)]],                               \
      const device int &num_kv_heads [[buffer(9)]],                            \
      const device int &has_key [[buffer(10)]],                                \
      uint token_idx [[threadgroup_position_in_grid]],                         \
      uint tid [[thread_position_in_threadgroup]],                             \
      uint threads_per_tg [[threads_per_threadgroup]]);

instantiate_rotary_embedding(float);
instantiate_rotary_embedding(half);
instantiate_rotary_embedding(bfloat16_t);
