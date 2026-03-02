#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("rotary_embedding(Tensor positions, Tensor! query,"
          "                 Tensor!? key, int head_size,"
          "                 Tensor cos_sin_cache, bool is_neox) -> ()");
#if defined(METAL_KERNEL)
  ops.impl("rotary_embedding", torch::kMPS, rotary_embedding);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
