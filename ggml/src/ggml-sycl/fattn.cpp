//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-mma-f16.hpp"
#include "fattn-tile.hpp"
#include "fattn-vec.hpp"
#include "fattn-wmma-f16.hpp"
#include "fattn.hpp"

template <int DKQ, int DV, int ncols2>
static void ggml_sycl_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_sycl_info().devices[ggml_sycl_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (Q->ne[1] <= 8/ncols2) {
            ggml_sycl_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_sycl_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 32/ncols2) {
        ggml_sycl_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_sycl_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_sycl_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_sycl_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_sycl_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_sycl_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_sycl_flash_attn_ext_mma_f16(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            GGML_ASSERT(gqa_ratio % 16 == 0);
            ggml_sycl_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) { \
        ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);  \
        return;                                                          \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

static void ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_SYCL_FA_ALL_QUANTS
    printf("zjy ggml_sycl_flash_attn_ext_vec 1\n");
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#else
    printf("zjy ggml_sycl_flash_attn_ext_vec 2\n");
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    printf("zjy ggml_sycl_get_best_fattn_kernel1 device=%d\n", device);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    const int cc = ggml_sycl_info().devices[device].cc;

    // TODO: temporary until support is extended
    //       https://github.com/ggml-org/llama.cpp/pull/16148#issuecomment-3343525206
    if (K->ne[1] % FATTN_KQ_STRIDE != 0) {
        printf("zjy ggml_sycl_get_best_fattn_kernel 20\n");
        return BEST_FATTN_KERNEL_NONE;
    }

    switch (K->ne[0]) {
        case  64:
        case 128:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                printf("zjy ggml_sycl_get_best_fattn_kernel 2\n");
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case  80:
        case  96:
        case 112:
            if (V->ne[0] != K->ne[0]) {
                printf("zjy ggml_sycl_get_best_fattn_kernel 3\n");
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!ggml_sycl_should_use_wmma_fattn(cc)) {
                printf("zjy ggml_sycl_get_best_fattn_kernel 4\n");
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                printf("zjy ggml_sycl_get_best_fattn_kernel 5\n");
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 16 != 0) {
                printf("zjy ggml_sycl_get_best_fattn_kernel 6\n");
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            printf("zjy ggml_sycl_get_best_fattn_kernel 7\n");
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_SYCL_FA_ALL_QUANTS
    if (K->type != V->type) {
        printf("zjy ggml_sycl_get_best_fattn_kernel 8\n");
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            printf("zjy ggml_sycl_get_best_fattn_kernel 9\n");
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
            break;
        default:
            printf("zjy ggml_sycl_get_best_fattn_kernel 10\n");
            return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        printf("zjy ggml_sycl_get_best_fattn_kernel 11\n");
        return BEST_FATTN_KERNEL_NONE;
    }

    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0;

        // Use kernels specialized for small batch sizes if possible:
    if (Q->ne[1] <= 8 && can_use_vector_kernel) {
        printf("zjy ggml_sycl_get_best_fattn_kernel 12\n");
        return BEST_FATTN_KERNEL_VEC;
    }

    // For large batch sizes, use the WMMA kernel if possible:
    if (ggml_sycl_should_use_wmma_fattn(cc)) {
        printf("zjy ggml_sycl_get_best_fattn_kernel 13\n");
        return BEST_FATTN_KERNEL_WMMA_F16;
    }
    printf("zjy ggml_sycl_get_best_fattn_kernel 14\n");
    // If there is no suitable kernel for tensor cores or small batch sizes, use the generic kernel for large batch sizes:
    return BEST_FATTN_KERNEL_TILE;
}

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);
    switch (ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst)) {
        case BEST_FATTN_KERNEL_NONE:
            printf("zjy ggml_sycl_flash_attn_ext 1\n");
            GGML_ABORT("fatal error");
        case BEST_FATTN_KERNEL_TILE:
        printf("zjy ggml_sycl_flash_attn_ext 2\n");
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            printf("zjy ggml_sycl_flash_attn_ext 3\n");
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_WMMA_F16:
            printf("zjy ggml_sycl_flash_attn_ext 4\n");
            ggml_sycl_flash_attn_ext_wmma_f16(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MMA_F16:
            printf("zjy ggml_sycl_flash_attn_ext 5\n");
            ggml_sycl_flash_attn_ext_mma_f16(ctx, dst);
            break;
    }
}

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
