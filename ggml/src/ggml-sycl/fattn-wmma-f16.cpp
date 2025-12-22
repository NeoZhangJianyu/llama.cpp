// Old and deprecated WMMA FlashAttention implementation.
// It is still needed for Volta since the memory layout of NVIDIA tensor cores changed with Turing.
// Long-term the WMMA code should be replaced with a dedicated Volta implementation.

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "sycl/half_type.hpp"
#include "fattn-common.hpp"
#include "fattn-wmma-f16.hpp"

#ifdef GGML_USE_WMMA_FATTN
#if !defined(GGML_USE_HIP)
#include <mma.h>
#if defined(GGML_USE_MUSA)
namespace wmma = mtmusa::wmma;
#else // GGML_USE_MUSA

#endif // GGML_USE_MUSA
#elif defined(GGML_USE_HIP)
#include <rocwmma/rocwmma.hpp>
namespace wmma = rocwmma;
#endif // !defined(GGML_USE_HIP)
#endif // GGML_USE_WMMA_FATTN

// D == head size, VKQ_stride == num VKQ rows calculated in parallel:
template <int D,
          int ncols,
          int nwarps,
          int VKQ_stride,
          typename KQ_acc_t,
          bool use_logit_softcap>
// SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
static void flash_attn_ext_f16(const char* Q,
                               const char* K,
                               const char* V,
                               const char* mask,
                               const char* sinks,
                               const int* KV_max,
                               float* dst,
                               sycl::float2* dst_meta,
                               const float scale,
                               const float max_bias,
                               const float m0,
                               const float m1,
                               const uint32_t n_head_log2,
                               const float logit_softcap,
                               const int32_t ne00,
                               const int32_t ne01,
                               const int32_t ne02,
                               const int32_t ne03,
                               const int32_t nb01,
                               const int32_t nb02,
                               const int32_t nb03,
                               const int32_t ne10,
                               const int32_t ne11,
                               const int32_t ne12,
                               const int32_t ne13,
                               const int32_t nb11,
                               const int32_t nb12,
                               const int64_t nb13,
                               const int32_t nb21,
                               const int32_t nb22,
                               const int64_t nb23,
                               const int32_t ne31,
                               const int32_t ne32,
                               const int32_t ne33,
                               const int32_t nb31,
                               const int32_t nb32,
                               const int64_t nb33,
                               const sycl::nd_item<3>& item_ct1,
                               uint8_t* unused_lsm) {
  // auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();

#if defined(FLASH_ATTN_AVAILABLE) && (DPCT_COMPATIBILITY_TEMP == GGML_SYCL_CC_VOLTA || \
                                      (defined(GGML_HIP_ROCWMMA_FATTN) && defined(GGML_USE_WMMA_FATTN)))
    // Skip unused kernel variants for faster compilation:
    if (use_logit_softcap && !(D == 128 || D == 256)) {
        return;
    }

    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    constexpr int warp_size = ggml_sycl_get_physical_warp_size();

    const int ic0 = ncols*item_ct1.get_group(2); // Index of the first Q/QKV column to work on.

    static_assert(D <= FATTN_KQ_STRIDE, "D must be <= FATTN_KQ_STRIDE.");
    static_assert(ncols == 8 || ncols % 16 == 0, "ncols must be 8 or a multiple of 16.");
    constexpr int frag_m = ncols == 8 ? 32 : 16;
    constexpr int frag_n = ncols == 8 ?  8 : 16;
    static_assert(D % frag_m == 0, "If ncols == 8 then D % frag_m must be 0.");
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, half, wmma::row_major> frag_a_K;
    typedef wmma::fragment<wmma::matrix_a,    frag_m, frag_n, 16, half, wmma::col_major> frag_a_V;
    typedef wmma::fragment<wmma::matrix_b,    frag_m, frag_n, 16, half, wmma::col_major> frag_b;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, KQ_acc_t>                      frag_c_KQ;
    typedef wmma::fragment<wmma::accumulator, frag_m, frag_n, 16, half>                          frag_c_VKQ;

    constexpr int KQ_stride_tc  = nwarps*frag_m; // Number of KQ rows calculated in parallel.
    constexpr int VKQ_ratio = KQ_stride_tc/VKQ_stride; // Number of parallel VKQ accumulators needed to keep all warps busy.
    static_assert(VKQ_ratio <= nwarps, "VKQ_ratio must be <= nwarps.");

    // Pad internal representation of KQ, KQV to reduce shared memory bank conflicts:
    constexpr int D_padded = D + 8;
    constexpr int kqs_padded = FATTN_KQ_STRIDE + 8;
    constexpr int kqar = sizeof(KQ_acc_t)/sizeof(half);

    const int sequence = item_ct1.get_group(0) / ne02;
    const int head = item_ct1.get_group(0) - sequence*ne02;
    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float * Q_f    = (const float *) (Q    + nb03* sequence         + nb02* head              + nb01*ic0);
    const half  * K_h    = (const half  *) (K    + nb13* sequence         + nb12*(head / gqa_ratio));
    const half  * V_h    = (const half  *) (V    + nb13* sequence         + nb12*(head / gqa_ratio)); // K and V have same shape
    const half  * maskh  = (const half  *) (mask + nb33*(sequence % ne33)                           + nb31*ic0);
    const half2 * mask2  = (const half2 *)  maskh;
    const float * sinksf = (const float *) sinks;

    const int stride_Q  = nb01 / sizeof(float);
    const int stride_KV = nb11 / sizeof(half);

    const float slopef = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);
    const half  slopeh = __float2half(slopef);
    const half2 slope2 = make_half2(slopef, slopef);

    const half2 logit_softcap_2 = make_half2(logit_softcap, logit_softcap);

    frag_b Q_b[D/16][ncols/frag_n];

    // A single buffer for temporarily holding tiles of KQ and VKQ parts:
    constexpr int mem_KQ = ncols*kqs_padded*kqar;
    constexpr int mem_VKQ_parts = VKQ_ratio*ncols*D_padded;
    // __shared__ half KQ[mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts];
    // __shared__ half VKQ[ncols*D_padded]; // Accumulator for final VKQ slice.
    size_t lsm_size1 = mem_KQ >= mem_VKQ_parts ? mem_KQ : mem_VKQ_parts*sizeof(half);
    size_t lsm_size2 = ncols*D_padded*sizeof(half);
    size_t local_share_mem_size = lsm_size1+lsm_size2;
    syclex::work_group_static<char[local_share_mem_size]> lsm;
    half *KQ = (half*) &lsm;
    half *VKQ = (half*) ((char*)&lsm + lsm_size1);

    float * KQ_f = (float *) KQ;
    half2 * KQ2 = (half2 *) KQ;

    float    KQ_rowsum_f[ncols/nwarps] = {0.0f};
    float       KQ_max_f[ncols/nwarps];
    float KQ_max_scale_f[ncols/nwarps] = {0.0f};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_f[j] = -FLT_MAX/2.0f;
    }

    half2    KQ_rowsum_h2[ncols/nwarps] = {{0.0f, 0.0f}};
    half2       KQ_max_h2[ncols/nwarps];
    half2 KQ_max_scale_h2[ncols/nwarps] = {{0.0f, 0.0f}};

#pragma unroll
    for (int j = 0; j < ncols/nwarps; ++j) {
        KQ_max_h2[j] = make_half2(-HALF_MAX_HALF, -HALF_MAX_HALF);
    }

    half2 * VKQ2 = (half2 *) VKQ;
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + item_ct1.get_local_id(1);
#pragma unroll
        for (int i0 = 0; i0 < D/2; i0 += warp_size) {
            const int i = i0 + item_ct1.get_local_id(2);
            if (i0 + warp_size > D/2 && i >= D/2) {
                break;
            }
            VKQ2[j*(D_padded/2) + i] = make_half2(0.0f, 0.0f);
        }
    }

    // Convert Q to half and apply scale, temporarily store in KQ:
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j = j0 + item_ct1.get_local_id(1);
#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + item_ct1.get_local_id(2);
            if (i0 + warp_size > D && i >= D) {
                break;
            }
            KQ[j*D_padded + i] = ic0 + j < ne01 ? Q_f[j*stride_Q + i] * scale : 0.0f;
        }
    }

    __syncthreads();

    // Load Q into tensor core fragments/registers since it will be used frequently:
#pragma unroll
    for (int i0 = 0; i0 < D; i0 += 16) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
            wmma::load_matrix_sync(Q_b[i0/16][j0/frag_n], KQ + j0*D_padded + i0, D_padded);
        }
    }

    __syncthreads();

    // Iterate over ne11 == previous tokens:
    const int k_VKQ_max = KV_max ? KV_max[sequence * item_ct1.get_group_range(2) + item_ct1.get_group(2)] : ne11;
    for (int k_VKQ_0 = item_ct1.get_group(1)*FATTN_KQ_STRIDE; k_VKQ_0 < k_VKQ_max; k_VKQ_0 += item_ct1.get_group_range(1)*FATTN_KQ_STRIDE) {
        // Calculate tile of KQ:
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < FATTN_KQ_STRIDE; i_KQ_0 += KQ_stride_tc) {
            frag_c_KQ KQ_c[ncols/frag_n];
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                wmma::fill_fragment(KQ_c[j], static_cast<KQ_acc_t>(0.0f));
            }
#pragma unroll
            for (int k_KQ_0 = 0; k_KQ_0 < D; k_KQ_0 += 16) {
                frag_a_K K_a;
                wmma::load_matrix_sync(K_a, K_h + int64_t(k_VKQ_0 + i_KQ_0 + frag_m*item_ct1.get_local_id(1))*stride_KV + k_KQ_0, stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    wmma::mma_sync(KQ_c[j], K_a, Q_b[k_KQ_0/16][j], KQ_c[j]);
                }
            }
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync((KQ_acc_t *) KQ + j0*kqs_padded + i_KQ_0 + frag_m*item_ct1.get_local_id(1), KQ_c[j0/frag_n], kqs_padded, wmma::mem_col_major);
            }
        }

        __syncthreads();

        // Calculate softmax for each KQ column using the current max. value.
        // The divisor is stored in KQ_rowsum and will be applied at the end.
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + item_ct1.get_local_id(1);

            if (std::is_same<KQ_acc_t, float>::value) {
                float KQ_f_tmp[FATTN_KQ_STRIDE / warp_size];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + item_ct1.get_local_id(2);

                    KQ_f_tmp[k0/warp_size] = KQ_f[j*kqs_padded + k];

                    if (use_logit_softcap) {
                        KQ_f_tmp[k0/warp_size] = logit_softcap*tanhf(KQ_f_tmp[k0/warp_size]);
                    }
                }

                float KQ_max_new = KQ_max_f[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + item_ct1.get_local_id(2);

                    KQ_f_tmp[k0/warp_size] += mask ? __half2float(slopeh*maskh[j*(nb31/sizeof(half)) + k_VKQ_0 + k]) : 0.0f;
                    KQ_max_new = max(KQ_max_new, KQ_f_tmp[k0/warp_size]);
                }
                KQ_max_new = warp_reduce_max<warp_size>(KQ_max_new);

                const float diff = KQ_max_f[j0/nwarps] - KQ_max_new;
                KQ_max_scale_f[j0/nwarps] = expf(diff);
                if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                    KQ_max_scale_f[j0/nwarps] = 0.0f;
                }
                KQ_max_f[j0/nwarps] = KQ_max_new;

                float KQ_rowsum_add = 0.0f;
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += warp_size) {
                    const int k = k0 + item_ct1.get_local_id(2);

                    const float diff = KQ_f_tmp[k0/warp_size] - KQ_max_f[j0/nwarps];
                    KQ_f_tmp[k0/warp_size] = expf(diff);
                    if (diff <= SOFTMAX_FTZ_THRESHOLD) {
                        KQ_f_tmp[k0/warp_size] = 0.0f;
                    }
                    KQ_rowsum_add += KQ_f_tmp[k0/warp_size];
                    KQ[j*(kqar*kqs_padded) + k] = KQ_f_tmp[k0/warp_size];
                }
                KQ_rowsum_add = warp_reduce_sum<warp_size>(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_f[j0/nwarps] = KQ_max_scale_f[j0/nwarps]*KQ_rowsum_f[j0/nwarps] + KQ_rowsum_add;
            } else {
                half2 KQ2_tmp[FATTN_KQ_STRIDE/(2*warp_size)];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += warp_size) {
                    const int k = k0 + item_ct1.get_local_id(2);

                    KQ2_tmp[k0/warp_size] = KQ2[j*(kqs_padded/2) + k];

                    if (use_logit_softcap) {
                        // There is no dedicated tangens hyperbolicus function for half2.
                        KQ2_tmp[k0/warp_size] = h2exp(KQ2_tmp[k0/warp_size]*make_half2(2.0f, 2.0f));
                        KQ2_tmp[k0/warp_size] = (KQ2_tmp[k0/warp_size] - make_half2(1.0f, 1.0f))
                                               /(KQ2_tmp[k0/warp_size] + make_half2(1.0f, 1.0f));

                        KQ2_tmp[k0/warp_size] *= logit_softcap_2;
                    }
                }

                half2 KQ_max_new = KQ_max_h2[j0/nwarps];
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += warp_size) {
                    const int k = k0 + item_ct1.get_local_id(2);

                    KQ2_tmp[k0/warp_size] += mask ? slope2*mask2[(j*ne11 + k_VKQ_0)/2 + k] : make_half2(0.0f, 0.0f);
                    KQ_max_new = ggml_sycl_hmax2(KQ_max_new, KQ2_tmp[k0/warp_size]);
                }
                KQ_max_new = __half2half2(warp_reduce_max<warp_size>(ggml_sycl_hmax(__low2half(KQ_max_new), __high2half(KQ_max_new))));
                const half2 diff = KQ_max_h2[j0/nwarps] - KQ_max_new;
                KQ_max_scale_h2[j0/nwarps] = h2exp(diff);
                const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                *((uint32_t *) &KQ_max_scale_h2[j0/nwarps]) &= ftz_mask;
                KQ_max_h2[j0/nwarps] = KQ_max_new;

                half2 KQ_rowsum_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int k0 = 0; k0 < FATTN_KQ_STRIDE/2; k0 += warp_size) {
                    const int k = k0 + item_ct1.get_local_id(2);

                    const half2 diff = KQ2_tmp[k0/warp_size] - KQ_max_h2[j0/nwarps];
                    KQ2_tmp[k0/warp_size] = h2exp(diff);
                    const uint32_t ftz_mask = __hgt2_mask(diff, make_half2(SOFTMAX_FTZ_THRESHOLD, SOFTMAX_FTZ_THRESHOLD));
                    *((uint32_t *) &KQ2_tmp[k0/warp_size]) &= ftz_mask;
                    KQ_rowsum_add += KQ2_tmp[k0/warp_size];
                    KQ2[j*(kqs_padded/2) + k] = KQ2_tmp[k0/warp_size];
                }
                KQ_rowsum_add = warp_reduce_sum<warp_size>(KQ_rowsum_add);

                // Scale previous KQ_rowsum to account for a potential increase in KQ_max:
                KQ_rowsum_h2[j0/nwarps] = KQ_max_scale_h2[j0/nwarps]*KQ_rowsum_h2[j0/nwarps] + KQ_rowsum_add;
            }
        }

        __syncthreads();

        frag_b KQ_b[FATTN_KQ_STRIDE/(VKQ_ratio*16)][ncols/frag_n];
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += frag_n) {
#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (item_ct1.get_local_id(1) % VKQ_ratio)*16;
                wmma::load_matrix_sync(
                    KQ_b[k0/(VKQ_ratio*16)][j0/frag_n],
                    KQ + j0*(kqar*kqs_padded) + k,
                    kqar*kqs_padded);
            }
        }

        frag_c_VKQ VKQ_c[D/VKQ_stride][ncols/frag_n];
#pragma unroll
        for (int i_VKQ_0 = 0; i_VKQ_0 < D; i_VKQ_0 += VKQ_stride) {
#pragma unroll
            for (int j = 0; j < ncols/frag_n; ++j) {
                wmma::fill_fragment(VKQ_c[i_VKQ_0/VKQ_stride][j], static_cast<half>(0.0f));
            }

#pragma unroll
            for (int k0 = 0; k0 < FATTN_KQ_STRIDE; k0 += VKQ_ratio*16) {
                const int k = k0 + (item_ct1.get_local_id(1) % VKQ_ratio)*16;

                frag_a_V v_a;
                wmma::load_matrix_sync(v_a, V_h + int64_t(k_VKQ_0 + k)*stride_KV + i_VKQ_0 + frag_m*(item_ct1.get_local_id(1)/VKQ_ratio), stride_KV);
#pragma unroll
                for (int j = 0; j < ncols/frag_n; ++j) {
                    wmma::mma_sync(VKQ_c[i_VKQ_0/VKQ_stride][j], v_a, KQ_b[k0/(VKQ_ratio*16)][j], VKQ_c[i_VKQ_0/VKQ_stride][j]);
                }
            }
        }

        __syncthreads();

        const int offset_k = (item_ct1.get_local_id(1) % VKQ_ratio) * (ncols*D_padded);
#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += VKQ_stride) {
#pragma unroll
            for (int j0 = 0; j0 < ncols; j0 += frag_n) {
                wmma::store_matrix_sync(
                    KQ + offset_k + j0*D_padded + i_KQ_0 + frag_m*(item_ct1.get_local_id(1)/VKQ_ratio),
                    VKQ_c[i_KQ_0/VKQ_stride][j0/frag_n],
                    D_padded, wmma::mem_col_major);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + item_ct1.get_local_id(1);

            half2 VKQ_scale;
            if (std::is_same<KQ_acc_t, float>::value) {
                VKQ_scale = make_half2(KQ_max_scale_f[j0/nwarps], KQ_max_scale_f[j0/nwarps]);
            } else {
                VKQ_scale = KQ_max_scale_h2[j0/nwarps];
            }

#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                const int i = i0 + item_ct1.get_local_id(2);
                if (i0 + warp_size > D/2 && i >= D/2) {
                    break;
                }

                half2 VKQ_add = make_half2(0.0f, 0.0f);
#pragma unroll
                for (int l = 0; l < VKQ_ratio; ++l) {
                    VKQ_add += KQ2[l*(ncols*D_padded/2) + j*(D_padded/2) + i];
                }
                VKQ2[j*(D_padded/2) + i] = VKQ_scale*VKQ2[j*(D_padded/2) + i] + VKQ_add;
            }
        }

        __syncthreads();
    }

    // Apply attention sinks
    if (sinksf && item_ct1.get_group(1) == 0) {
        const float sinkf = sinksf[head];
        const half  sinkh = __float2half(sinkf);

#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + item_ct1.get_local_id(1);

            if (std::is_same<KQ_acc_t, float>::value) {
                float kqmax_new = fmaxf(KQ_max_f[j0/nwarps], sinkf);

                const float KQ_max_scale = expf(KQ_max_f[j0/nwarps] - kqmax_new);
                KQ_max_f[j0/nwarps] = kqmax_new;

                KQ_rowsum_f[j0/nwarps] = KQ_rowsum_f[j0/nwarps] * KQ_max_scale + expf(sinkf - KQ_max_f[j0/nwarps]);

                const half2 scale_h2 = make_half2(KQ_max_scale, KQ_max_scale);
#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + item_ct1.get_local_id(2);
                    if (i0 + warp_size > D/2 && i >= D/2) break;
                    VKQ2[j*(D_padded/2) + i] *= scale_h2;
                }
            } else {
                half kqmax_old = __low2half(KQ_max_h2[j0/nwarps]);
                half kqmax_new = fmaxf(kqmax_old, sinkh);
                KQ_max_h2[j0/nwarps] = __half2half2(kqmax_new);

                const half  KQ_max_scale_h = hexp(kqmax_old - kqmax_new);
                const half2 KQ_max_scale   = __half2half2(KQ_max_scale_h);

                KQ_rowsum_h2[j0/nwarps] = KQ_rowsum_h2[j0/nwarps] * KQ_max_scale;
                const half val = hexp(sinkh - kqmax_new);
                KQ_rowsum_h2[j0/nwarps].x = __hadd(KQ_rowsum_h2[j0/nwarps].x, val);

#pragma unroll
                for (int i0 = 0; i0 < D/2; i0 += warp_size) {
                    const int i = i0 + item_ct1.get_local_id(2);
                    if (i0 + warp_size > D/2 && i >= D/2) break;
                    VKQ2[j*(D_padded/2) + i] *= KQ_max_scale;
                }
            }
        }

        __syncthreads();
    }
#pragma unroll
    for (int j0 = 0; j0 < ncols; j0 += nwarps) {
        const int j_VKQ = j0 + item_ct1.get_local_id(1);
        if (ic0 + j_VKQ >= ne01) {
            return;
        }

        float KQ_rowsum_j;
        if (std::is_same<KQ_acc_t, float>::value) {
            KQ_rowsum_j = KQ_rowsum_f[j0/nwarps];
        } else {
            KQ_rowsum_j = __low2float(KQ_rowsum_h2[j0/nwarps]) + __high2float(KQ_rowsum_h2[j0/nwarps]);
        }

        const int j_dst_unrolled = ((sequence*ne01 + ic0 + j_VKQ)*ne02 + head)*item_ct1.get_group_range(1) + item_ct1.get_group(1);

#pragma unroll
        for (int i0 = 0; i0 < D; i0 += warp_size) {
            const int i = i0 + item_ct1.get_local_id(2);
            if (i0 + warp_size > D && i >= D) {
                break;
            }
            float dst_val = VKQ[j_VKQ*D_padded + i];
            if (item_ct1.get_group_range(1) == 1) {
                dst_val /= KQ_rowsum_j;
            }
            dst[j_dst_unrolled*D + i] = dst_val;
        }

        if (item_ct1.get_group_range(1) == 1 || item_ct1.get_local_id(2) != 0) {
            continue;
        }

        sycl::float2 dst_meta_val;
        if (std::is_same<KQ_acc_t, float>::value) {
            dst_meta_val.x = KQ_max_f[j0/nwarps];
        } else {
            dst_meta_val.x = __low2float(KQ_max_h2[j0/nwarps]);
        }
        dst_meta_val.y = KQ_rowsum_j;
        dst_meta[j_dst_unrolled] = dst_meta_val;
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
#endif // defined(FLASH_ATTN_AVAILABLE) && (__SYCL_ARCH__ == GGML_SYCL_CC_VOLTA || (defined(GGML_HIP_ROCWMMA_FATTN) && defined(GGML_USE_WMMA_FATTN)))
}

/*
DPCT1109:41: Recursive functions cannot be called in SYCL device code. You need to adjust the code.
*/
constexpr int get_max_power_of_2(int x) {
    /*
    DPCT1109:42: Recursive functions cannot be called in SYCL device code. You need to adjust the code.
    */
    return x % 2 == 0 ? 2 * get_max_power_of_2(x / 2) : 1;
}

static_assert(get_max_power_of_2(1) == 1, "Test failed.");
static_assert(get_max_power_of_2(2) == 2, "Test failed.");
static_assert(get_max_power_of_2(4) == 4, "Test failed.");
static_assert(get_max_power_of_2(6) == 2, "Test failed.");

// Number of VKQ rows calculated in parallel:
constexpr int get_VKQ_stride(int D, int nwarps, int frag_m) {
    /*
    DPCT1109:43: Recursive functions cannot be called in SYCL device code. You need to adjust the code.
    */
    return (get_max_power_of_2(D / frag_m) < nwarps ? get_max_power_of_2(D / frag_m) : nwarps) * frag_m;
}

static_assert(get_VKQ_stride(128, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride(128, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride(128, 4, 32) == 128, "Test failed.");
static_assert(get_VKQ_stride( 64, 1, 32) ==  32, "Test failed.");
static_assert(get_VKQ_stride( 64, 2, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 64, 4, 32) ==  64, "Test failed.");
static_assert(get_VKQ_stride( 80, 1, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 2, 16) ==  16, "Test failed.");
static_assert(get_VKQ_stride( 80, 4, 16) ==  16, "Test failed.");

template <int D, int cols_per_block, typename KQ_acc_t>
void ggml_sycl_flash_attn_ext_wmma_f16_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;

    constexpr int nwarps = 4;

    constexpr int frag_m = cols_per_block == 8 && D % 32 == 0 ? 32 : 16;
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    // fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
      constexpr bool use_logit_softcap = false;
      launch_fattn<D, cols_per_block, 1,
                   flash_attn_ext_f16<D, cols_per_block, nwarps,
                                      get_VKQ_stride(D, nwarps, frag_m),
                                      KQ_acc_t, false>>(
          ctx, dst, nwarps, 0, FATTN_KQ_STRIDE, true, true, false, warp_size);

    } else {
      constexpr bool use_logit_softcap = true;

      launch_fattn<D, cols_per_block, 1,
                   flash_attn_ext_f16<D, cols_per_block, nwarps,
                                      get_VKQ_stride(D, nwarps, frag_m),
                                      KQ_acc_t, use_logit_softcap>>(
          ctx, dst, nwarps, 0, FATTN_KQ_STRIDE, true, true, false, warp_size);
    }
}

void ggml_sycl_flash_attn_ext_wmma_f16(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const ggml_tensor * Q   = dst->src[0];

    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);
    const int warp_size = ggml_sycl_info().devices[ctx.device].warp_size;

    if (prec != GGML_PREC_DEFAULT) {
        if (Q->ne[1] <= 32 || Q->ne[0] > 128) {
            constexpr int cols_per_block = 16;
            switch (Q->ne[0]) {
                case 64:
                    ggml_sycl_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_sycl_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_sycl_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_sycl_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_sycl_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                    break;
                case 256:
                    ggml_sycl_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
                    break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        } else {
            constexpr int cols_per_block = 32;
            switch (Q->ne[0]) {
                case 64:
                    ggml_sycl_flash_attn_ext_wmma_f16_case< 64, cols_per_block, float>(ctx, dst);
                    break;
                case 80:
                    ggml_sycl_flash_attn_ext_wmma_f16_case< 80, cols_per_block, float>(ctx, dst);
                    break;
                case 96:
                    ggml_sycl_flash_attn_ext_wmma_f16_case< 96, cols_per_block, float>(ctx, dst);
                    break;
                case 112:
                    ggml_sycl_flash_attn_ext_wmma_f16_case<112, cols_per_block, float>(ctx, dst);
                    break;
                case 128:
                    ggml_sycl_flash_attn_ext_wmma_f16_case<128, cols_per_block, float>(ctx, dst);
                    break;
                // case 256:
                //     ggml_sycl_flash_attn_ext_wmma_f16_case<256, cols_per_block, float>(ctx, dst);
                //     break;
                default:
                    GGML_ABORT("fatal error");
                    break;
            }
        }
        return;
    }

#if !defined(GGML_USE_HIP)
    if (Q->ne[1] <= 8 && Q->ne[0] % warp_size == 0) {
        constexpr int cols_per_block = 8;
        switch (Q->ne[0]) {
            case 64:
                ggml_sycl_flash_attn_ext_wmma_f16_case<64, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 96:
                ggml_sycl_flash_attn_ext_wmma_f16_case<96, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_wmma_f16_case<128, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_wmma_f16_case<256, cols_per_block, sycl::half>(ctx, dst);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }
#endif // !defined(GGML_USE_HIP)

    if (Q->ne[1] <= 32) {
        constexpr int cols_per_block = 16;
        switch (Q->ne[0]) {
            case 64:
                ggml_sycl_flash_attn_ext_wmma_f16_case<64, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 80:
                ggml_sycl_flash_attn_ext_wmma_f16_case<80, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 96:
                ggml_sycl_flash_attn_ext_wmma_f16_case<96, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 112:
                ggml_sycl_flash_attn_ext_wmma_f16_case<112, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_wmma_f16_case<128, cols_per_block, sycl::half>(ctx, dst);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_wmma_f16_case<256, cols_per_block, sycl::half>(ctx, dst);
                break;
            default:
                GGML_ABORT("fatal error");
                break;
        }
        return;
    }

    constexpr int cols_per_block = 32;
    switch (Q->ne[0]) {
        case 64:
            ggml_sycl_flash_attn_ext_wmma_f16_case<64, cols_per_block, sycl::half>(ctx, dst);
            break;
        case 80:
            ggml_sycl_flash_attn_ext_wmma_f16_case<80, cols_per_block, sycl::half>(ctx, dst);
            break;
        case 96:
            ggml_sycl_flash_attn_ext_wmma_f16_case<96, cols_per_block, sycl::half>(ctx, dst);
            break;
        case 112:
            ggml_sycl_flash_attn_ext_wmma_f16_case<112, cols_per_block, sycl::half>(ctx, dst);
            break;
        case 128:
            ggml_sycl_flash_attn_ext_wmma_f16_case<128, cols_per_block, sycl::half>(ctx, dst);
            break;
        case 256:
            ggml_sycl_flash_attn_ext_wmma_f16_case<256, cols_per_block, sycl::half>(ctx, dst);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
