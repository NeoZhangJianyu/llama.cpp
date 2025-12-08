#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"

// WMMA flash attention requires FP16 matrix instructions to be available for ggml code.
static bool ggml_sycl_should_use_wmma_fattn(const int cc) {
    //todo detect the Intel GPU XMX capablity.
    return true;
}

void ggml_sycl_flash_attn_ext_wmma_f16(ggml_backend_sycl_context & ctx, ggml_tensor * dst);
