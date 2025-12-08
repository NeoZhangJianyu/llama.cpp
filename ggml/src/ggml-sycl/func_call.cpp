#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include <vector>
#include <iostream>

using namespace sycl;
using namespace dpct;

int main() {
    const int N = 10; // Size of vectors

    // Initialize host data
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Create a SYCL queue to submit work to a device
    sycl::queue q;

    // Create SYCL buffers from host data
    sycl::buffer<int, 1> buffer_a(a.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> buffer_b(b.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> buffer_c(c.data(), sycl::range<1>(N));

    // Submit a command group to the queue
    q.submit([&](sycl::handler &h) {
        // Create accessors to the buffers
        sycl::accessor a_acc(buffer_a, h, sycl::read_only);
        sycl::accessor b_acc(buffer_b, h, sycl::read_only);
        sycl::accessor c_acc(buffer_c, h, sycl::write_only);

        // Define the kernel for vector addition
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            c_acc[idx] = a_acc[idx] + b_acc[idx];
        });
    });

    // When the buffer_c goes out of scope, data is copied back to host
    // (or explicitly wait for completion using q.wait_and_throw() if needed)

    // Print the results
    std::cout << "Vector C (A + B):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "C[" << i << "] = " << c[i] << std::endl;
    }

    return 0;
}
