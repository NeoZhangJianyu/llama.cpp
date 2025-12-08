#define DPCT_COMPAT_RT_VERSION 12000
// Simplified API for asynchronous data loading.

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"

static __dpct_inline__ unsigned int ggml_sycl_cvta_generic_to_shared(void * generic_ptr) {
#ifdef CP_ASYNC_AVAILABLE
    return __cvta_generic_to_shared(generic_ptr);
#else
    GGML_UNUSED(generic_ptr);
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload> static __dpct_inline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#    if DPCT_COMPAT_RT_VERSION >= 11040
    if (preload == 256) {
        /*
        DPCT1053:8: Migration of device assembly code is not supported.
        */
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;" : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        /*
        DPCT1053:9: Migration of device assembly code is not supported.
        */
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        /*
        DPCT1053:10: Migration of device assembly code is not supported.
        */
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;" : : "r"(dst), "l"(src));
    } else
#endif // SYCLRT_VERSION >= 11040
    {
        /*
        DPCT1137:11: ASM instruction "cp.async" is asynchronous copy, currently it is migrated to synchronous copy operation. You may need to adjust the code to tune the performance.
        */
        *(((uint32_t *) (uintptr_t) dst)) = *(((uint32_t *) (uintptr_t) src));
        if (16 > 4)
            *(((uint32_t *) (uintptr_t) dst) + 1) = *(((uint32_t *) (uintptr_t) src) + 1);
        if (16 > 8)
            *(((uint32_t *) (uintptr_t) dst) + 2) = *(((uint32_t *) (uintptr_t) src) + 2);
        if (16 > 12)
            *(((uint32_t *) (uintptr_t) dst) + 3) = *(((uint32_t *) (uintptr_t) src) + 3);
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
#endif // CP_ASYNC_AVAILABLE
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __dpct_inline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    /*
    DPCT1026:12: The call to "cp.async.wait_all;" was removed because current "cp.async" is migrated to synchronous copy operation. You may need to adjust the code to tune the performance.
    */

#else
    //NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
