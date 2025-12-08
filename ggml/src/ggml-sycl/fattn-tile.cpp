#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include "fattn-wmma-f16.hpp"
#include <cmath>
#include <float.h>
namespace syclex = sycl::ext::oneapi::experimental;

// kq_stride == number of KQ rows to process per iteration
// kq_nbatch == number of K columns to load in parallel for KQ calculation

static int fattn_tile_get_kq_stride_host(const int D, const int ncols, const int cc, const int warp_size) {
    if (fast_fp16_available(cc)) {
        switch (D) {
            case 64:
            case 128:
            case 256:
                return ncols <= 16 ? 128 : 64;
            default:
                GGML_ABORT("fatal error");
                return -1;
        }
    }
    switch (D) {
        case 64:
            return ncols <= 16 ? 128 : 64;
        case 128:
            return ncols <= 16 ? 64 : 32;
        case 256:
            return 32;
        default:
            GGML_ABORT("fatal error");
            return -1;
    }
    GGML_UNUSED(warp_size);
}

static constexpr int fattn_tile_get_kq_stride_device(int D, int ncols, int warp_size) {
#ifdef FAST_FP16_AVAILABLE
    switch (D) {
        case 64:
        case 128:
        case 256:
            return ncols <= 16 ? 128 : 64;
        default:
            return -1;
    }
#else
    switch (D) {
        case 64:
            return ncols <= 16 ? 128 : 64;
        case 128:
            return ncols <= 16 ? 64 : 32;
        case 256:
            return 32;
        default:
            return -1;
    }
#endif // FAST_FP16_AVAILABLE
    GGML_UNUSED_VARS(ncols, warp_size);
}

static constexpr int fattn_tile_get_kq_nbatch_device(int D, int ncols, int warp_size) {
#ifdef FAST_FP16_AVAILABLE
    switch (D) {
        case 64:
            return 64;
        case 128:
        case 256:
            return 128;
        default:
            return -1;
    }
#else
    switch (D) {
        case 64:
            return 64;
        case 128:
            return 128;
        case 256:
            return ncols <= 16 ? 128 : 64;
        default:
            return -1;
    }
#endif // FAST_FP16_AVAILABLE

    GGML_UNUSED_VARS(ncols, warp_size);
}

static int fattn_tile_get_nthreads_host(const int cc, const int ncols) {
    return 256;
    GGML_UNUSED_VARS(cc, ncols);
}

static constexpr int fattn_tile_get_nthreads_device(int ncols) {
    return 256;
    GGML_UNUSED(ncols);
}

static constexpr int fattn_tile_get_occupancy_device(int ncols) {
    return ncols <= 16 ? 3 : 2;
    GGML_UNUSED(ncols);
}

template <int D, int ncols, bool use_logit_softcap>  // D == head size
/*
DPCT1110:35: The total declared local variable size in device function flash_attn_tile exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/

static void flash_attn_tile(const char *  Q,
                            const char *  K,
                            const char *  V,
                            const char *  mask,
                            const char *  sinks,
                            const int *  KV_max,
                            float *  dst,
                            sycl::float2 *  dst_meta,
                            const float          scale,
                            const float          max_bias,
                            const float          m0,
                            const float          m1,
                            const uint32_t       n_head_log2,
                            const float          logit_softcap,
                            const int32_t        ne00,
                            const int32_t        ne01,
                            const int32_t        ne02,
                            const int32_t        ne03,
                            const int32_t        nb01,
                            const int32_t        nb02,
                            const int32_t        nb03,
                            const int32_t        ne10,
                            const int32_t        ne11,
                            const int32_t        ne12,
                            const int32_t        ne13,
                            const int32_t        nb11,
                            const int32_t        nb12,
                            const int64_t        nb13,
                            const int32_t        nb21,
                            const int32_t        nb22,
                            const int64_t        nb23,
                            const int32_t        ne31,
                            const int32_t        ne32,
                            const int32_t        ne33,
                            const int32_t        nb31,
                            const int32_t        nb32,
                            const int64_t        nb33) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

#ifdef FLASH_ATTN_AVAILABLE

    // Skip unused kernel variants for faster compilation:
#ifdef GGML_USE_WMMA_FATTN
    return;
#endif // GGML_USE_WMMA_FATTN

    if (use_logit_softcap && !(D == 128 || D == 256)) {
        GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
            max_bias, m0, m1, n_head_log2, logit_softcap,
            ne00, ne01, ne02, ne03,
                  nb01, nb02, nb03,
            ne10, ne11, ne12, ne13,
                  nb11, nb12, nb13,
                  nb21, nb22, nb23,
                  ne31, ne32, ne33,
                  nb31, nb32, nb33);
        return;
    }

    constexpr int warp_size = 32;
    constexpr int nwarps    = fattn_tile_get_nthreads_device(ncols) / warp_size;
    constexpr int kq_stride = fattn_tile_get_kq_stride_device(D, ncols, warp_size);
    static_assert(kq_stride % warp_size == 0, "kq_stride not divisable by warp_size.");
    constexpr int kq_nbatch = fattn_tile_get_kq_nbatch_device(D, ncols, warp_size);
    static_assert(kq_nbatch % (2*warp_size) == 0, "bad kq_nbatch");

    // In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = item_ct1.get_group(2) * ncols;  // Index of the Q/QKV column to work on.

    const int           sequence  = item_ct1.get_group(0) / ne02;
    const int           head      = item_ct1.get_group(0) - sequence * ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f    = (const float *) (Q    + nb03* sequence         + nb02* head              + nb01*ic0);
    const sycl::half2 * K_h2      = (const sycl::half2 *) (K + nb13 * sequence + nb12 * (head / gqa_ratio));
    const sycl::half2 * V_h2 =
        (const sycl::half2 *) (V + nb13 * sequence + nb12 * (head / gqa_ratio));  // K and V have same shape
    const sycl::half * maskh  = (const sycl::half *) (mask + nb33 * (sequence % ne33) + nb31 * ic0);
    const float * sinksf = (const float *) (sinks);

    const int stride_KV2 = nb11 / sizeof(sycl::half2);

    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    constexpr int cpy_nb = ggml_sycl_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int cpw = ncols/nwarps; // cols per warp

    // softmax_iter_j == number of KQ columns for which to calculate softmax in parallel.
    // KQ is originall 2D but uses a Z-shaped memory pattern for larger reads/writes.
#ifdef FAST_FP16_AVAILABLE
    constexpr int softmax_iter_j = cpw < 2*cpy_ne ? cpw : 2*cpy_ne;
    // __shared__ half  KQ[ncols/softmax_iter_j][kq_stride][softmax_iter_j];
    // __shared__ half2 Q_tmp[ncols][D/2];
    // __shared__ half2 KV_tmp[kq_stride * (kq_nbatch/2 + cpy_ne)]; // Padded to avoid memory bank conflicts.

    constexpr int KQ_d1 = int(ncols/softmax_iter_j);
    constexpr int KQ_d2 = kq_stride;
    constexpr int KQ_d3 = softmax_iter_j;
    constexpr size_t lsm_size1 = KQ_d1 * KQ_d2 * KQ_d2*sizeof(half);

    constexpr int Q_tmp_d1 = ncols;
    constexpr int Q_tmp_d2 = int(D/2);
    constexpr size_t lsm_size2 = Q_tmp_d1 * Q_tmp_d2 * sizeof(sycl::half2);

    constexpr int KV_tmp_d1 = kq_stride * (kq_nbatch/2 + cpy_ne);
    constexpr size_t lsm_size3 = KV_tmp_d1 * sizeof(sycl::half2);

    constexpr size_t local_share_mem_size = lsm_size1 + lsm_size2 + lsm_size3;
    syclex::work_group_static<char[local_share_mem_size]> lsm;
    half (*KQ)[KQ_d1][KQ_d2][KQ_d3] = (half (*)[KQ_d1][KQ_d2][KQ_d3])&lsm;
    sycl::half2 (*Q_tmp)[Q_tmp_d1][Q_tmp_d2] = (sycl::half2 (*)[Q_tmp_d1][Q_tmp_d2]) (&lsm+lsm_size1);
    sycl::half2 (*KV_tmp)[KV_tmp_d1] = (sycl::half2 (*)[KV_tmp_d1]) (&lsm+lsm_size1+lsm_size2);

    sycl::half2 VKQ[cpw][D / (2 * warp_size)] = { { { 0.0f, 0.0f } } };
#else
    constexpr int softmax_iter_j = cpw < 1*cpy_ne ? cpw : 1*cpy_ne;

    constexpr int KQ_d1 = int(ncols/softmax_iter_j);
    constexpr int KQ_d2 = kq_stride;
    constexpr int KQ_d3 = softmax_iter_j;
    constexpr size_t lsm_size1 = KQ_d1 * KQ_d2 * KQ_d2*sizeof(float);

    constexpr int Q_tmp_d1 = ncols;
    constexpr int Q_tmp_d2 = D;
    constexpr size_t lsm_size2 = Q_tmp_d1 * Q_tmp_d2 * sizeof(float);

    constexpr int KV_tmp_d1 = kq_stride * (kq_nbatch + cpy_ne);
    constexpr size_t lsm_size3 = KV_tmp_d1 * sizeof(float);

    constexpr size_t local_share_mem_size = lsm_size1 + lsm_size2 + lsm_size3;
    syclex::work_group_static<char[local_share_mem_size]> lsm;
    // __shared__ float KQ[ncols/softmax_iter_j][kq_stride][softmax_iter_j];
    // __shared__ float Q_tmp[ncols][D];
    // __shared__ float KV_tmp[kq_stride * (kq_nbatch + cpy_ne)]; // Padded to avoid memory bank conflicts.

    float (*KQ)[KQ_d1][KQ_d2][KQ_d3] = (float (*)[KQ_d1][KQ_d2][KQ_d3])&lsm;
    float (*Q_tmp)[Q_tmp_d1][Q_tmp_d2] = (float (*)[Q_tmp_d1][Q_tmp_d2]) (&lsm+lsm_size1);
    float (*KV_tmp)[KV_tmp_d1] = (float (*)[KV_tmp_d1]) (&lsm+lsm_size1+lsm_size2);

    sycl::float2 VKQ[cpw][D/(2*warp_size)] = {{{0.0f, 0.0f}}};
#endif // FAST_FP16_AVAILABLE
    static_assert(cpw % softmax_iter_j == 0, "bad softmax_iter_j");

    float KQ_max[cpw];
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        KQ_max[j0/nwarps] = -FLT_MAX/2.0f;
    }
    float KQ_sum[cpw] = {0.0f};

    // Load Q data, convert to FP16 if fast.
#pragma unroll
    for (int j0 = 0; j0 < cpw; ++j0) {
        const int j = j0 + item_ct1.get_local_id(1) * cpw;

        constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
            float tmp_f[cpy_ne_D] = {0.0f};
            if (ic0 + j < ne01) {
                ggml_sycl_memcpy_1<sizeof(tmp_f)>(
                    tmp_f, &Q_f[j * (nb01 / sizeof(float)) + i0 + item_ct1.get_local_id(2) * cpy_ne_D]);
            }

#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp_f[i1] *= scale;
            }

#ifdef FAST_FP16_AVAILABLE
            sycl::half2 tmp_h2[cpy_ne_D / 2];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; i1 += 2) {
                tmp_h2[i1/2] = make_half2(tmp_f[i1 + 0], tmp_f[i1 + 1]);
            }
            ggml_sycl_memcpy_1<sizeof(tmp_h2)>(&Q_tmp[j][i0 / 2 + item_ct1.get_local_id(2) * (cpy_ne_D / 2)], tmp_h2);
#else
            ggml_sycl_memcpy_1<sizeof(tmp_f)> (&Q_tmp[j][i0   + item_ct1.get_local_id(2) * cpy_ne_D],    tmp_f);
#endif // FAST_FP16_AVAILABLE
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main loop over KV cache:
    const int k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    for (int k_VKQ_0 = item_ct1.get_group(1) * kq_stride; k_VKQ_0 < k_VKQ_max;
         k_VKQ_0 += item_ct1.get_group_range(1) * kq_stride) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float KQ_max_new[cpw];
#pragma unroll
        for (int j = 0; j < cpw; ++j) {
            KQ_max_new[j] = KQ_max[j];
        }

        float KQ_acc[kq_stride/warp_size][cpw] = {{0.0f}}; // Accumulators for KQ matrix multiplication.

        // KQ = K @ Q matrix multiplication:
#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += kq_nbatch) {
#pragma unroll
            for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += nwarps) {
                const int i_KQ = i_KQ_0 + item_ct1.get_local_id(1);

#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_kqnb = cpy_ne < kq_nbatch/(2*warp_size) ? cpy_ne : kq_nbatch/(2*warp_size);
#pragma unroll
                for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch/2; k_KQ_1 += warp_size*cpy_ne_kqnb) {
                    ggml_sycl_memcpy_1<cpy_ne_kqnb * 4>(
                        &KV_tmp[i_KQ * (kq_nbatch / 2 + cpy_ne) + k_KQ_1 + item_ct1.get_local_id(2) * cpy_ne_kqnb],
                        &K_h2[int64_t(k_VKQ_0 + i_KQ) * stride_KV2 + k_KQ_0 / 2 + k_KQ_1 +
                              item_ct1.get_local_id(2) * cpy_ne_kqnb]);
                }
#else
                constexpr int cpy_ne_kqnb = cpy_ne < kq_nbatch/warp_size ? cpy_ne : kq_nbatch/warp_size;
#pragma unroll
                for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch; k_KQ_1 += warp_size*cpy_ne_kqnb) {
                    sycl::half2 tmp_h2[cpy_ne_kqnb/2];
                    ggml_sycl_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, &K_h2[int64_t(k_VKQ_0 + i_KQ)*stride_KV2 + k_KQ_0/2 + k_KQ_1/2 + item_ct1.get_local_id(2)*(cpy_ne_kqnb/2)]);

                    sycl::float2 tmp_f2[cpy_ne_kqnb/2];
#pragma unroll
                    for (int k_KQ_2 = 0; k_KQ_2 < cpy_ne_kqnb/2; ++k_KQ_2) {
                        tmp_f2[k_KQ_2] = __half22float2(tmp_h2[k_KQ_2]);
                    }
                    ggml_sycl_memcpy_1<sizeof(tmp_f2)>(
                        &KV_tmp[i_KQ*(kq_nbatch + cpy_ne) + k_KQ_1 + item_ct1.get_local_id(2)*cpy_ne_kqnb], tmp_f2);
                }
#endif // FAST_FP16_AVAILABLE
            }

            /*
            DPCT1118:37: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
            for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch/2; k_KQ_1 += cpy_ne) {
                sycl::half2 K_k[kq_stride / warp_size][cpy_ne];
                sycl::half2 Q_k[cpw][cpy_ne];
#else
#pragma unroll
            for (int k_KQ_1 = 0; k_KQ_1 < kq_nbatch; k_KQ_1 += cpy_ne) {
                float K_k[kq_stride/warp_size][cpy_ne];
                float Q_k[cpw][cpy_ne];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
                    const int i_KQ = i_KQ_0 + item_ct1.get_local_id(2);

#ifdef FAST_FP16_AVAILABLE
                    ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/warp_size], &KV_tmp[i_KQ*(kq_nbatch/2 + cpy_ne) + k_KQ_1]);
#else
                    ggml_sycl_memcpy_1<cpy_nb>(&K_k[i_KQ_0/warp_size], &KV_tmp[i_KQ*(kq_nbatch   + cpy_ne) + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
                }
#pragma unroll
                for (int j_KQ_0 = 0; j_KQ_0 < cpw; ++j_KQ_0) {
                    const int j_KQ = j_KQ_0 + item_ct1.get_local_id(1) * cpw;

#ifdef FAST_FP16_AVAILABLE
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_k[j_KQ_0], &Q_tmp[j_KQ][k_KQ_0/2 + k_KQ_1]);
#else
                    ggml_sycl_memcpy_1<cpy_nb>(&Q_k[j_KQ_0], &Q_tmp[j_KQ][k_KQ_0   + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
                }

#pragma unroll
                for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
#pragma unroll
                    for (int j_KQ_0 = 0; j_KQ_0 < cpw; ++j_KQ_0) {
#pragma unroll
                        for (int k = 0; k < cpy_ne; ++k) {
                            ggml_sycl_mad(KQ_acc[i_KQ_0/warp_size][j_KQ_0], K_k[i_KQ_0/warp_size][k], Q_k[j_KQ_0][k]);
                        }
                    }
                }
            }

            if (k_KQ_0 + kq_nbatch < D) {
                /*
                DPCT1118:38: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
                */
                item_ct1.barrier(sycl::access::fence_space::local_space);  // Sync not needed on last iteration.
            }
        }

        // Apply logit softcap, mask, update KQ_max:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < kq_stride; i_KQ_0 += warp_size) {
            const int i_KQ = i_KQ_0 + item_ct1.get_local_id(2);

#pragma unroll
            for (int j_KQ_0 = 0; j_KQ_0 < cpw; ++j_KQ_0) {
                const int j_KQ = j_KQ_0 + item_ct1.get_local_id(1) * cpw;

                if (use_logit_softcap) {
                    KQ_acc[i_KQ_0 / warp_size][j_KQ_0] =
                        logit_softcap * sycl::tanh((float) KQ_acc[i_KQ_0 / warp_size][j_KQ_0]);
                }

                KQ_acc[i_KQ_0 / warp_size][j_KQ_0] +=
                    mask ? slope * sycl::vec<sycl::half, 1>(maskh[j_KQ * ne11 + k_VKQ_0 + i_KQ])
                                       .convert<float, sycl::rounding_mode::automatic>()[0] :
                           0.0f;

                KQ_max_new[j_KQ_0] = sycl::fmax((float) KQ_max_new[j_KQ_0], (float) KQ_acc[i_KQ_0 / warp_size][j_KQ_0]);
            }
        }

        /*
        DPCT1118:36: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
        */
        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Calculate KQ softmax, write to shared KQ buffer, re-scale VKQ accumulators:
#pragma unroll
        for (int j0 = 0; j0 < cpw; j0 += softmax_iter_j) {
#ifdef FAST_FP16_AVAILABLE
            sycl::half tmp[kq_stride / warp_size][softmax_iter_j];
#else
            float tmp[kq_stride/warp_size][softmax_iter_j];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
            for (int j1 = 0; j1 < softmax_iter_j; ++j1) {
                KQ_max_new[j0+j1] = warp_reduce_max<warp_size>(KQ_max_new[j0+j1]);
                const float KQ_max_scale = sycl::native::exp((float) (KQ_max[j0 + j1] - KQ_max_new[j0 + j1]));
                KQ_max[j0+j1] = KQ_max_new[j0+j1];

                float KQ_sum_add = 0.0f;
#pragma unroll
                for (int i0 = 0; i0 < kq_stride; i0 += warp_size) {
                    const float val = sycl::native::exp((float) (KQ_acc[i0 / warp_size][j0 + j1] - KQ_max[j0 + j1]));
                    KQ_sum_add += val;
                    tmp[i0/warp_size][j1] = val;
                }
                KQ_sum[j0+j1] = KQ_sum[j0+j1]*KQ_max_scale + KQ_sum_add;

#ifdef FAST_FP16_AVAILABLE
                const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    VKQ[j0+j1][i0/warp_size] *= KQ_max_scale_h2;
                }
#else
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    VKQ[j0+j1][i0/warp_size].x() *= KQ_max_scale;
                    VKQ[j0+j1][i0/warp_size].y() *= KQ_max_scale;
                }
#endif // FAST_FP16_AVAILABLE
            }

#pragma unroll
            for (int i0 = 0; i0 < kq_stride; i0 += warp_size) {
                const int i = i0 + item_ct1.get_local_id(2);

                ggml_sycl_memcpy_1<sizeof(tmp[0])>(
                    KQ[j0 / softmax_iter_j + item_ct1.get_local_id(1) * (cpw / softmax_iter_j)][i],
                    tmp[i0 / warp_size]);
            }
        }

        // VKQ = V @ KQ matrix multiplication:
        constexpr int V_cols_per_iter = kq_stride*kq_nbatch / D; // Number of V columns that fit in SRAM for K.
        static_assert(kq_stride % V_cols_per_iter == 0, "bad V_cols_per_iter");
#pragma unroll
        for (int k0 = 0; k0 < kq_stride; k0 += V_cols_per_iter) {
#pragma unroll
            for (int k1 = 0; k1 < V_cols_per_iter; k1 += nwarps) {
                const int k_tile = k1 + item_ct1.get_local_id(1);

#ifdef FAST_FP16_AVAILABLE
                constexpr int cpy_ne_D = cpy_ne < D/(2*warp_size) ? cpy_ne : D/(2*warp_size);
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
                    ggml_sycl_memcpy_1<cpy_ne_D * 4>(
                        &KV_tmp[k_tile * (D / 2) + i0 + item_ct1.get_local_id(2) * cpy_ne_D],
                        &V_h2[int64_t(k_VKQ_0 + k0 + k_tile) * stride_KV2 + i0 + item_ct1.get_local_id(2) * cpy_ne_D]);
                }
#else
                constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
                    sycl::half2 tmp_h2[cpy_ne_D/2];
                    ggml_sycl_memcpy_1<sizeof(tmp_h2)>(
                        tmp_h2, &V_h2[int64_t(k_VKQ_0 + k0 + k_tile)*stride_KV2 + i0/2 + item_ct1.get_local_id(2)*(cpy_ne_D/2)]);

                    sycl::float2 tmp_f2[cpy_ne_D/2];
#pragma unroll
                    for (int i1 = 0; i1 < cpy_ne_D/2; ++i1) {
                        tmp_f2[i1] = __half22float2(tmp_h2[i1]);
                    }
                    ggml_sycl_memcpy_1<sizeof(tmp_f2)>(
                        &KV_tmp[k_tile*D + i0 + item_ct1.get_local_id(2)*cpy_ne_D], tmp_f2);
                }
#endif // FAST_FP16_AVAILABLE
            }

            /*
            DPCT1118:39: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);

#ifdef FAST_FP16_AVAILABLE
#pragma unroll
            for (int k1 = 0; k1 < V_cols_per_iter; ++k1) {
                sycl::half2 V_k[(D / 2) / warp_size];
                sycl::half2 KQ_k[cpw];

                constexpr int cpy_ne_D = cpy_ne/2 < (D/2)/warp_size ? cpy_ne/2 : (D/2)/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
                    ggml_sycl_memcpy_1<cpy_ne_D * 4>(&V_k[i0 / warp_size],
                                                     &KV_tmp[k1 * (D / 2) + i0 + item_ct1.get_local_id(2) * cpy_ne_D]);
                }
#pragma unroll
                for (int j0 = 0; j0 < cpw; j0 += softmax_iter_j) {
                    const int j = j0 / softmax_iter_j + item_ct1.get_local_id(1) * (cpw / softmax_iter_j);

                    sycl::half tmp[softmax_iter_j];
                    ggml_sycl_memcpy_1<softmax_iter_j * sizeof(sycl::half)>(&tmp, KQ[j][k0 + k1]);
#pragma unroll
                    for (int j1 = 0; j1 < softmax_iter_j; ++j1) {
                        KQ_k[j0 + j1] = sycl::half2(tmp[j1]);
                    }
                }

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
#pragma unroll
                    for (int j0 = 0; j0 < cpw; ++j0) {
                        VKQ[j0][i0/warp_size] += V_k[i0/warp_size]*KQ_k[j0];
                    }
                }
            }
#else
#pragma unroll
            for (int k1 = 0; k1 < V_cols_per_iter; ++k1) {
                sycl::float2 V_k[(D/2)/warp_size];
                float  KQ_k[cpw];

                constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;
#pragma unroll
                for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
                    ggml_sycl_memcpy_1<cpy_ne_D*4>(&V_k[i0/(2*warp_size)], &KV_tmp[k1*D + i0 + item_ct1.get_local_id(2)*cpy_ne_D]);
                }
#pragma unroll
                for (int j0 = 0; j0 < cpw; j0 += softmax_iter_j) {
                    const int j = j0/softmax_iter_j + item_ct1.get_local_id(1)*(cpw/softmax_iter_j);

                    ggml_sycl_memcpy_1<softmax_iter_j*sizeof(float)>(
                        &KQ_k[j0], &KQ[j][k0 + k1]);
                }

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
#pragma unroll
                    for (int j0 = 0; j0 < cpw; ++j0) {
                        VKQ[j0][i0/warp_size].x() += V_k[i0/warp_size].x()*KQ_k[j0];
                        VKQ[j0][i0/warp_size].y() += V_k[i0/warp_size].y()*KQ_k[j0];
                    }
                }
            }
#endif // FAST_FP16_AVAILABLE

            /*
            DPCT1118:40: SYCL group functions and algorithms must be encountered in converged control flow. You may need to adjust the code.
            */
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }
    }


    // Attention sink: adjust running max and sum once per head
    if (sinksf && item_ct1.get_group(1) == 0) {
        const float sink = sinksf[head];

#pragma unroll
        for (int j0 = 0; j0 < cpw; ++j0) {
            float KQ_max_new_j = sycl::fmax((float) KQ_max[j0], sink);
            KQ_max_new_j = warp_reduce_max<warp_size>(KQ_max_new_j);

            const float KQ_max_scale = sycl::native::exp((float) (KQ_max[j0] - KQ_max_new_j));
            KQ_max[j0] = KQ_max_new_j;

            const float val = sycl::native::exp((float) (sink - KQ_max[j0]));
            KQ_sum[j0] = KQ_sum[j0] * KQ_max_scale;
            if (item_ct1.get_local_id(2) == 0) {
                KQ_sum[j0] += val;
            }

#ifdef FAST_FP16_AVAILABLE
            const sycl::half2 KQ_max_scale_h2 = sycl::half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0][i0/warp_size] *= KQ_max_scale_h2;
            }
#else
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                VKQ[j0][i0/warp_size].x() *= KQ_max_scale;
                VKQ[j0][i0/warp_size].y() *= KQ_max_scale;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < cpw; ++j_VKQ_0) {
        KQ_sum[j_VKQ_0] = warp_reduce_sum<warp_size>(KQ_sum[j_VKQ_0]);
    }
    if (item_ct1.get_group_range(1) == 1) {
#pragma unroll
        for (int j_VKQ_0 = 0; j_VKQ_0 < cpw; ++j_VKQ_0) {
#ifdef FAST_FP16_AVAILABLE
            const sycl::half2 KQ_sum_j_inv = make_half2(1.0f / KQ_sum[j_VKQ_0], 1.0f / KQ_sum[j_VKQ_0]);
#pragma unroll
            for (int i = 0; i < (D/2)/warp_size; ++i) {
                VKQ[j_VKQ_0][i] *= KQ_sum_j_inv;
            }
#else
            const float KQ_sum_j_inv = 1.0f/KQ_sum[j_VKQ_0];
#pragma unroll
            for (int i = 0; i < (D/2)/warp_size; ++i) {
                VKQ[j_VKQ_0][i].x() *= KQ_sum_j_inv;
                VKQ[j_VKQ_0][i].y() *= KQ_sum_j_inv;
            }
#endif // FAST_FP16_AVAILABLE
        }
    }

    // Write back results:
#pragma unroll
    for (int j_VKQ_0 = 0; j_VKQ_0 < cpw; ++j_VKQ_0) {
        const int j_VKQ = j_VKQ_0 + item_ct1.get_local_id(1) * cpw;

        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        const int j_dst_unrolled =
            ((sequence * ne01 + ic0 + j_VKQ) * ne02 + head) * item_ct1.get_group_range(1) + item_ct1.get_group(1);

#ifdef FAST_FP16_AVAILABLE
        constexpr int cpy_ne_D = cpy_ne/2 < (D/2)/warp_size ? cpy_ne/2 : (D/2)/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
            sycl::float2 tmp[cpy_ne_D];
#pragma unroll
            for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
                tmp[i1] = VKQ[j_VKQ_0][i0 / warp_size + i1].template convert<float, sycl::rounding_mode::automatic>();
            }
            ggml_sycl_memcpy_1<sizeof(tmp)>(
                &dst[j_dst_unrolled * D + 2 * i0 + item_ct1.get_local_id(2) * (2 * cpy_ne_D)], tmp);
        }
#else
        constexpr int cpy_ne_D = cpy_ne < D/warp_size ? cpy_ne : D/warp_size;
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size*cpy_ne_D) {
            ggml_sycl_memcpy_1<cpy_ne_D*4>(
                &dst[j_dst_unrolled*D + i0 + item_ct1.get_local_id(2)*cpy_ne_D], &VKQ[j_VKQ_0][i0/(2*warp_size)]);
        }
#endif // FAST_FP16_AVAILABLE

        if (item_ct1.get_group_range(1) != 1 && item_ct1.get_local_id(2) == 0) {
            dst_meta[j_dst_unrolled] = make_float2(KQ_max[j_VKQ_0], KQ_sum[j_VKQ_0]);
        }
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
              ne31, ne32, ne33,
              nb31, nb32, nb33);
#endif // FLASH_ATTN_AVAILABLE
}

template <int D, bool use_logit_softcap>
static void launch_fattn_tile_switch_ncols(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    const int id        = ggml_sycl_get_device();
    const int cc        = ggml_sycl_info().devices[id].cc;
    const int warp_size = 32;

    constexpr size_t nbytes_shared = 0;

#ifdef GGML_USE_HIP
    if constexpr (D <= 128) {
        if (Q->ne[1] > 32) {
            constexpr int cols_per_block = 64;
            const int nwarps = fattn_tile_get_nthreads_host(cc, cols_per_block) / warp_size;
            fattn_kernel_t fattn_kernel = flash_attn_tile<D, cols_per_block, use_logit_softcap>;
            const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, cc, warp_size);
            launch_fattn<D, cols_per_block, 1>
                (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
            return;
        }
    }
#endif // GGML_USE_HIP

    if (Q->ne[1] > 16) {
        constexpr int cols_per_block = 32;
        const int nwarps = fattn_tile_get_nthreads_host(cc, cols_per_block) / warp_size;
        fattn_kernel_t fattn_kernel   = flash_attn_tile<D, cols_per_block, use_logit_softcap>;
        const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, cc, warp_size);
        launch_fattn<D, cols_per_block, 1>
            (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
        return;
    }

    constexpr int cols_per_block = 16;
    const int nwarps = fattn_tile_get_nthreads_host(cc, cols_per_block) / warp_size;
    fattn_kernel_t fattn_kernel   = flash_attn_tile<D, cols_per_block, use_logit_softcap>;
    const int kq_stride = fattn_tile_get_kq_stride_host(D, cols_per_block, cc, warp_size);
    launch_fattn<D, cols_per_block, 1>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared, kq_stride, true, true, false, warp_size);
}

template <bool use_logit_softcap>
static void launch_fattn_tile_switch_head_size(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            launch_fattn_tile_switch_ncols< 64, use_logit_softcap>(ctx, dst);
        } break;
        case 128: {
            launch_fattn_tile_switch_ncols<128, use_logit_softcap>(ctx, dst);
        } break;
        case 256: {
            launch_fattn_tile_switch_ncols<256, use_logit_softcap>(ctx, dst);
        } break;
        default: {
            GGML_ABORT("Unsupported head size");
        } break;
    }
}

void ggml_sycl_flash_attn_ext_tile(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        launch_fattn_tile_switch_head_size<use_logit_softcap>(ctx, dst);
    } else {
        constexpr bool use_logit_softcap = true;
        launch_fattn_tile_switch_head_size<use_logit_softcap>(ctx, dst);
    }
}
