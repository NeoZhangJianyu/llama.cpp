#define DPCT_COMPAT_RT_VERSION 12000
#pragma once
// This file contains primitives that expose the tensor core PTX instructions for SYCL code.
// The primitives can be used in a similar way as the nvsycl::wmma interface but with a well-defined memory layout.
// The documentation for the PTX instructions can be found under:
//   https://docs.nvidia.com/sycl/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction
//
// Like with nvsycl::wmma there are three types of matrix tiles: A, B, and C with A @ B = C.
// A is a row-major matrix with shape M x K.
// B is a column-major matrix with shape K x N.
// C is a column-major matrix with shape M x N.
// A, B, and C are represented using the same fundamental data type: a row-major matrix with I rows and J columns.
// Note that J is measured in physical 32 bit elements instead of logical elements.
// The methods get_i and get_j can be used to get the physical 32 bit index of the lth element of a thread within a tile.
// All matrix tiles have ne physical 32 bit elements per warp.
//
// As described in the PTX documentation, all pointers for load_ldmatrix must be to shared memory and aligned to 16 bytes.
// The API in this file also assumes that the pointers for load_generic are aligned to 16 bytes, unaligned pointers are considered undefined behavior.

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include <cmath>

#if DPCT_COMPAT_RT_VERSION >= 11080

static __dpct_inline__ int ggml_sycl_movmatrix(const int x) {
    int ret = 0;

#ifdef TURING_MMA_AVAILABLE
    /*
    DPCT1053:13: Migration of device assembly code is not supported.
    */
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(ret) : "r"(x));
#else
    GGML_UNUSED(x);
#endif // defined(TURING_MMA_AVAILABLE)
    return ret;
}

#else

static __device__ __forceinline__ int ggml_sycl_movmatrix(const int x) {
    // Imagine transposing row-major matrix to column-major matrix.
    const int src_i_low  = 2 * (threadIdx.x % 4);
    const int src_i_high = src_i_low + 1;
    const int src_j      = threadIdx.x / 4;

    const int src_laneid_low  = src_i_low  * 4 + src_j / 2;
    const int src_laneid_high = src_i_high * 4 + src_j / 2;

    const int shift_low  = ((src_j + 0) % 2) * 16;
    const int shift_high = ((src_j + 1) % 2) * 16;

    const int ret_low  = (__shfl_sync(0xFFFFFFFF, x, src_laneid_low,  WARP_SIZE) >> shift_low)  & 0x0000FFFF;
    const int ret_high = (__shfl_sync(0xFFFFFFFF, x, src_laneid_high, WARP_SIZE) << shift_high) & 0xFFFF0000;

    return ret_low | ret_high;
}

#endif // SYCLRT_VERSION >= 11080

static __dpct_inline__ sycl::half2 ggml_sycl_movmatrix(const sycl::half2 x) {
    sycl::half2 ret;
    *((int *) &ret) = ggml_sycl_movmatrix(*((const int *) &x));
    return ret;
}

namespace ggml_sycl_mma {

    template <int I_, int J_, typename T>
    struct tile {
        static constexpr int I  = I_;
        static constexpr int J  = J_;

#if defined(GGML_USE_HIP)
        static constexpr int ne = I * J / 64;
        T x[ne] = {0};

        static __device__ __forceinline__ int get_i(const int l) {
            if constexpr (I == 64 && J == 2) { // Special tile size to load <16, 4> as <16, 8>
                return threadIdx.x % 16;
            } else if constexpr (I == 16 && J == 8) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 4) {
                return threadIdx.x % 32;
            } else if constexpr (I == 16 && J == 16) {
                return 4 * (threadIdx.x / 16) + l;
            } else if constexpr (I == 32 && J == 32) {
                return 4 * (threadIdx.x / 32) + 8 * (l / 4) + (l % 4);
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }

        static __device__ __forceinline__ int get_j(const int l) {
            if constexpr (I == 64 && J == 2) { // Special tile size to load <16, 4> as <16, 8>
                return (2 * ((threadIdx.x / 16) % 2) + l);
            } else if constexpr (I == 16 && J == 8) {
                return 2 * (threadIdx.x / 16) + l;
            } else if constexpr (I == 32 && J == 4) {
                return 2 * (threadIdx.x / 32) + l;
            } else if constexpr (I == 16 && J == 16) {
                return threadIdx.x % 16;
            } else if constexpr (I == 32 && J == 32) {
                return threadIdx.x % 32;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }
#else
        static constexpr int ne = I * J / 32;
        T x[ne] = {0};

        static __dpct_inline__ int get_i(const int l) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if constexpr (I == 8 && (J == 4 || J == 8)) {
                return item_ct1.get_local_id(2) / 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l / 2) * 8 + item_ct1.get_local_id(2) / 4;
            } else if constexpr (I == 16 && J == 16) {
                return ((l / 2) % 2) * 8 + item_ct1.get_local_id(2) / 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }

        static __dpct_inline__ int get_j(const int l) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if constexpr (I == 8 && J == 4) {
                return item_ct1.get_local_id(2) % 4;
            } else if constexpr (I == 8 && J == 8) {
                return 4 * l + item_ct1.get_local_id(2) % 4;
            } else if constexpr (I == 16 && J == 8) {
                return 2 * (item_ct1.get_local_id(2) % 4) + l % 2;
            } else if constexpr (I == 16 && J == 16) {
                return 8 * (l / 4) + 2 * (item_ct1.get_local_id(2) % 4) + l % 2;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }
#endif // defined(GGML_USE_HIP)
    };

    template <int I_, int J_> struct tile<I_, J_, sycl::half2> {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / WARP_SIZE;
        sycl::half2          x[ne] = {
            { 0.0f, 0.0f }
        };

        static __dpct_inline__ int get_i(const int l) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if constexpr (I == 8 && J == 8) {
                return item_ct1.get_local_id(2) / 4;
            } else if constexpr (I == 16 && J == 4) {
                return l * 8 + item_ct1.get_local_id(2) / 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l % 2) * 8 + item_ct1.get_local_id(2) / 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }

        static __dpct_inline__ int get_j(const int l) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if constexpr (I == 8 && J == 8) {
                return l * 4 + item_ct1.get_local_id(2) % 4;
            } else if constexpr (I == 16 && J == 4) {
                return item_ct1.get_local_id(2) % 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l / 2) * 4 + item_ct1.get_local_id(2) % 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }
    };

    template <int I_, int J_> struct tile<I_, J_, sycl::vec<sycl::ext::oneapi::bfloat16, 2>> {
        static constexpr int I  = I_;
        static constexpr int J  = J_;
        static constexpr int ne = I * J / WARP_SIZE;
        sycl::vec<sycl::ext::oneapi::bfloat16, 2> x[ne] = {
            { 0.0f, 0.0f }
        };

        static __dpct_inline__ int get_i(const int l) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if constexpr (I == 8 && J == 8) {
                return item_ct1.get_local_id(2) / 4;
            } else if constexpr (I == 16 && J == 4) {
                return l * 8 + item_ct1.get_local_id(2) / 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l % 2) * 8 + item_ct1.get_local_id(2) / 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }

        static __dpct_inline__ int get_j(const int l) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if constexpr (I == 8 && J == 8) {
                return l * 4 + item_ct1.get_local_id(2) % 4;
            } else if constexpr (I == 16 && J == 4) {
                return item_ct1.get_local_id(2) % 4;
            } else if constexpr (I == 16 && J == 8) {
                return (l / 2) * 4 + item_ct1.get_local_id(2) % 4;
            } else {
                static_assert(I == -1 && J == -1, "template specialization not implemented");
            }
        }
    };

    template <int I, int J>
    static __dpct_inline__ tile<I, J / 2, sycl::half2> get_half2(const tile<I, J, float> & tile_float) {
        tile<I, J / 2, sycl::half2> ret;
#pragma unroll
        for (int l0 = 0; l0 < tile_float.ne; l0 += 2) {
            ret.x[l0/2] = make_half2(tile_float.x[l0 + 0], tile_float.x[l0 + 1]);
        }
        return ret;
    }

    static __dpct_inline__ tile<8, 8, sycl::half2> get_transposed(const tile<16, 4, sycl::half2> & t) {
        tile<8, 8, sycl::half2> ret;
        ret.x[0] = ggml_sycl_movmatrix(t.x[0]);
        ret.x[1] = ggml_sycl_movmatrix(t.x[1]);

        return ret;
    }

    template <int I, int J, typename T>
    static __dpct_inline__ void load_generic(tile<I, J, T> & t, const T * __restrict__ xs0, const int stride) {
#if defined(AMD_MFMA_AVAILABLE)
        if constexpr (I == 64 && J == 2) { // Special tile size to load <16, 4> as <16, 8>
#pragma unroll
            for (int l = 0; l < t.ne; ++l) {
                t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
            }
        } else {
            int64_t * xi = (int64_t *) t.x;
            const int64_t * xs = (int64_t *) ((const int *) xs0 + (threadIdx.x % t.I) * stride + 2 * (threadIdx.x / t.I));
            xi[0] = xs[0];
        }
#else
#pragma unroll
        for (int l = 0; l < t.ne; ++l) {
            t.x[l] = xs0[t.get_i(l)*stride + t.get_j(l)];
        }
#endif // defined(AMD_MFMA_AVAILABLE)
    }

    template <typename T>
    static __dpct_inline__ void load_ldmatrix(tile<8, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        auto            item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int *           xi       = (int *) t.x;
        const int * xs       = (const int *) xs0 + (item_ct1.get_local_id(2) % t.I) * stride +
                         ((item_ct1.get_local_id(2) / t.I) * (t.J / 2)) % t.J;
    dpct::experimental::matrix::ldmatrix((uintptr_t) xs, &xi[0], &xi[1]);
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T>
    static __dpct_inline__ void load_ldmatrix(tile<16, 4, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        int * xi = (int *) t.x;
        const int * xs =
            (const int *) xs0 + (sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) % t.I) * stride;
    dpct::experimental::matrix::ldmatrix((uintptr_t) xs, &xi[0], &xi[1]);
#else
        load_generic(xs0, stride);
        GGML_UNUSED(t);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T>
    static __dpct_inline__ void load_ldmatrix(tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#if defined(TURING_MMA_AVAILABLE)
        auto            item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int *           xi       = (int *) t.x;
        const int * xs       = (const int *) xs0 + (item_ct1.get_local_id(2) % t.I) * stride +
                         (item_ct1.get_local_id(2) / t.I) * (t.J / 2);
    dpct::experimental::matrix::ldmatrix((uintptr_t) xs, &xi[0], &xi[1], &xi[2], &xi[3]);
#else
        load_generic(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    template <typename T>
    static __dpct_inline__ void load_ldmatrix_trans(tile<16, 8, T> & t, const T * __restrict__ xs0, const int stride) {
#ifdef TURING_MMA_AVAILABLE
        auto            item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int *           xi       = (int *) t.x;
        const int * xs       = (const int *) xs0 + (item_ct1.get_local_id(2) % t.I) * stride +
                         (item_ct1.get_local_id(2) / t.I) * (t.J / 2);
    dpct::experimental::matrix::ldmatrix((uintptr_t) xs, &xi[0], &xi[2], &xi[1], &xi[3], true);
#else
        GGML_UNUSED_VARS(t, xs0, stride);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 8, int> & D, const tile<16, 4, int> & A, const tile<8, 4, int> & B) {
#ifdef TURING_MMA_AVAILABLE
#    if DPCT_COMPATIBILITY_TEMP >= GGML_SYCL_CC_AMPERE
    {
        volatile int32_t *     d_mat_frag_ct1[4] = { &D.x[0], &D.x[1], &D.x[2], &D.x[3] };
        sycl::vec<uint32_t, 2> a_mat_frag_ct1(A.x[0], A.x[1]);
        sycl::vec<uint32_t, 1> b_mat_frag_ct1(B.x[0]);
        sycl::vec<int32_t, 4>  c_mat_frag_ct1(D.x[0], D.x[1], D.x[2], D.x[3]);
        dpct::experimental::matrix::mma<16, 8, 16, int8_t, int32_t>(reinterpret_cast<volatile void **>(d_mat_frag_ct1),
                                                                    &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
#endif // __SYCL_ARCH__ >= GGML_SYCL_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 8, int> & D, const tile<16, 8, int> & A, const tile<8, 8, int> & B) {
#ifdef TURING_MMA_AVAILABLE
#    if DPCT_COMPATIBILITY_TEMP >= GGML_SYCL_CC_AMPERE
    {
        volatile int32_t *     d_mat_frag_ct1[4] = { &D.x[0], &D.x[1], &D.x[2], &D.x[3] };
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(A.x[0], A.x[1], A.x[2], A.x[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(B.x[0], B.x[1]);
        sycl::vec<int32_t, 4>  c_mat_frag_ct1(D.x[0], D.x[1], D.x[2], D.x[3]);
        dpct::experimental::matrix::mma<16, 8, 32, int8_t, int32_t>(reinterpret_cast<volatile void **>(d_mat_frag_ct1),
                                                                    &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#else
        // On Turing m16n8k32 mma is not available, use 4x m8n8k16 mma instead:
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[0]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[1]), "r"(B.x[0]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[0]), "+r"(D.x[1])
            : "r"(A.x[2]), "r"(B.x[1]));
        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
            : "+r"(D.x[2]), "+r"(D.x[3])
            : "r"(A.x[3]), "r"(B.x[1]));
#endif // __SYCL_ARCH__ >= GGML_SYCL_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 4, sycl::half2> &       D,
                                    const tile<16, 8, sycl::half2> & A,
                                    const tile<8, 8, sycl::half2> &  B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#    if DPCT_COMPATIBILITY_TEMP >= GGML_SYCL_CC_AMPERE
        /*
        DPCT1053:14: Migration of device assembly code is not supported.
        */
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __SYCL_ARCH__ >= GGML_SYCL_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 8, sycl::half2> &       D,
                                    const tile<16, 8, sycl::half2> & A,
                                    const tile<16, 8, sycl::half2> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#    if DPCT_COMPATIBILITY_TEMP >= GGML_SYCL_CC_AMPERE
        /*
        DPCT1053:15: Migration of device assembly code is not supported.
        */
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[2]));
        /*
        DPCT1053:16: Migration of device assembly code is not supported.
        */
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]), "r"(Bxi[3]));
#else
        // On Turing m16n8k16 mma is not available, use 4x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[0]), "+r"(Dxi[1])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
            : "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __SYCL_ARCH__ >= GGML_SYCL_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 8, float> & D, const tile<16, 8, float> & A, const tile<8, 8, float> & B) {
#ifdef AMPERE_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
        /*
        DPCT1053:17: Migration of device assembly code is not supported.
        */
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, "
            "%2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // AMPERE_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 8, float> &             D,
                                    const tile<16, 8, sycl::half2> & A,
                                    const tile<8, 8, sycl::half2> &  B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#    if DPCT_COMPATIBILITY_TEMP >= GGML_SYCL_CC_AMPERE
    {
        volatile float *       d_mat_frag_ct1[4] = { &Dxi[0], &Dxi[1], &Dxi[2], &Dxi[3] };
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(Axi[0], Axi[1], Axi[2], Axi[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(Bxi[0], Bxi[1]);
        sycl::vec<float, 4>    c_mat_frag_ct1(Dxi[0], Dxi[1], Dxi[2], Dxi[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#else
        // On Turing m16n8k16 mma is not available, use 2x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[1]));
#endif // __SYCL_ARCH__ >= GGML_SYCL_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 8, float> &                                           D,
                                    const tile<16, 8, sycl::vec<sycl::ext::oneapi::bfloat16, 2>> & A,
                                    const tile<8, 8, sycl::vec<sycl::ext::oneapi::bfloat16, 2>> &  B) {
#ifdef AMPERE_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
        /*
        DPCT1053:18: Migration of device assembly code is not supported.
        */
        asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, "
            "%1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]));
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // AMPERE_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 16, float> &            D,
                                    const tile<16, 8, sycl::half2> & A,
                                    const tile<16, 8, sycl::half2> & B) {
#ifdef TURING_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        int       * Dxi = (int       *) D.x;
#    if DPCT_COMPATIBILITY_TEMP >= GGML_SYCL_CC_AMPERE
    {
        volatile float *       d_mat_frag_ct1[4] = { &Dxi[0], &Dxi[1], &Dxi[2], &Dxi[3] };
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(Axi[0], Axi[1], Axi[2], Axi[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(Bxi[0], Bxi[2]);
        sycl::vec<float, 4>    c_mat_frag_ct1(Dxi[0], Dxi[1], Dxi[2], Dxi[3]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
    {
        volatile float *       d_mat_frag_ct1[4] = { &Dxi[4], &Dxi[5], &Dxi[6], &Dxi[7] };
        sycl::vec<uint32_t, 4> a_mat_frag_ct1(Axi[0], Axi[1], Axi[2], Axi[3]);
        sycl::vec<uint32_t, 2> b_mat_frag_ct1(Bxi[1], Bxi[3]);
        sycl::vec<float, 4>    c_mat_frag_ct1(Dxi[4], Dxi[5], Dxi[6], Dxi[7]);
        dpct::experimental::matrix::mma<16, 8, 16, sycl::half, float>(
            reinterpret_cast<volatile void **>(d_mat_frag_ct1), &a_mat_frag_ct1, &b_mat_frag_ct1, &c_mat_frag_ct1);
    }
#else
        // On Turing m16n8k16 mma is not available, use 4x m8n8k8 mma instead:
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[0]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[0]), "+r"(Dxi[1]), "+r"(Dxi[2]), "+r"(Dxi[3])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[2]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
            : "r"(Axi[0]), "r"(Axi[1]), "r"(Bxi[1]));
        asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
            : "+r"(Dxi[4]), "+r"(Dxi[5]), "+r"(Dxi[6]), "+r"(Dxi[7])
            : "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[3]));
#endif // __SYCL_ARCH__ >= GGML_SYCL_CC_AMPERE
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // TURING_MMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<16, 16, int> &      D,
                                    const tile<16, 8, int> & A,
                                    const tile<16, 8, int> & B,
                                    const sycl::stream &     stream_ct1) {
#if defined(AMD_MFMA_AVAILABLE)
        using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
        int32x4_t * acc = (int32x4_t *) D.x;
#if defined(CDNA3)
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x32_i8(((int64_t *) A.x)[0],
                                                       ((int64_t *) B.x)[0],
                                                       acc[0],
                                                       0, 0, 0);
#elif defined(CDNA2) || defined(CDNA)
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(A.x[0],
                                                      B.x[0],
                                                      acc[0],
                                                      0, 0, 0);
        acc[0] = __builtin_amdgcn_mfma_i32_16x16x16i8(A.x[1],
                                                      B.x[1],
                                                      acc[0],
                                                      0, 0, 0);
#endif // defined(CDNA3)
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // AMD_MFMA_AVAILABLE
    }

    static __dpct_inline__ void mma(tile<32, 32, int> &      D,
                                    const tile<32, 4, int> & A,
                                    const tile<32, 4, int> & B,
                                    const sycl::stream &     stream_ct1) {
#if defined(AMD_MFMA_AVAILABLE)
        using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;
        int32x16_t * acc = (int32x16_t *) D.x;
#if defined(CDNA3)
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x16_i8(((int64_t *) A.x)[0],
                                                       ((int64_t *) B.x)[0],
                                                       acc[0],
                                                       0, 0, 0);
#elif defined(CDNA2) || defined(CDNA)
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(A.x[0],
                                                     B.x[0],
                                                     acc[0],
                                                     0, 0, 0);
        acc[0] = __builtin_amdgcn_mfma_i32_32x32x8i8(A.x[1],
                                                     B.x[1],
                                                     acc[0],
                                                     0, 0, 0);
#endif // defined(CDNA3)
#else
        GGML_UNUSED_VARS(D, A, B);
#endif // AMD_MFMA_AVAILABLE
    }
}
