#include <cstdio>
#include <alpaka/alpaka.hpp>

#include "config.h"
#include "workdivision.h"

const int DSIZE = 518;
const int RADIUS = 3;
const int BLOCK_SIZE = 32;
const int A_intval = 1;
const int B_intval = 2;

struct MatMulKernel {
    //Note that we assume A and B are square matrices throughout
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ in_A,
                                  T const* __restrict__ in_B,
                                  T* __restrict__ out,
                                  Vec2D size) const {
        for (auto ndindex : elements_with_stride_nd(acc, size)) {
            if (ndindex[0] < DSIZE && ndindex[1] < DSIZE) {

                //Apply matrix multiplication for element (ndindex[0],ndindex[1])
                int result = 0;
                for (int i=0; i<size[1]; i++) {
                    auto index_A = ndindex[0]*size[1] + i;
                    auto index_B = i*size[1] + ndindex[1];
                    result += in_A[index_A]*in_B[index_B];
                }
                out[ndindex[0]*size[1] + ndindex[1]] = result;
            }
        }
    }
};

struct StencilKernel {
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  T const* __restrict__ in,
                                  T* __restrict__ out,
                                  Vec2D size) const {
        for (auto ndindex : elements_with_stride_nd(acc, size)) {
            if ((ndindex[0] >= RADIUS && ndindex[0] < size[0]-RADIUS) && (ndindex[1] >= RADIUS && ndindex[1] < size[1]-RADIUS)) {
                int result = in[ndindex[0]*size[1] + ndindex[1]];
                for (int offset = 1; offset <= RADIUS; offset++){
                    result += in[(ndindex[0]+offset)*size[1]+ndindex[1]];
                    result += in[(ndindex[0]-offset)*size[1]+ndindex[1]];
                    result += in[ndindex[0]*size[1]+ndindex[1]+offset];
                    result += in[ndindex[0]*size[1]+ndindex[1]-offset];
                }
                out[ndindex[0]*size[1]+ndindex[1]] = result;
            }
        }
    }
};

int main() {
    //require at least one device
    std::size_t n = alpaka::getDevCount<Platform>();
    if (n==0) {
        exit(EXIT_FAILURE);
    }

    //use single host device
    Host host = alpaka::getDevByIdx<HostPlatform>(0u);
    std::cout << "Host:   " << alpaka::getName(host) << '\n';

    //use the first device
    Device device = alpaka::getDevByIdx<Platform>(0u);
    std::cout << "Device: " << alpaka::getName(device) << '\n';

    // 2D and linearized buffer size
    constexpr Vec2D m_ndsize = {DSIZE,DSIZE};
    constexpr size_t m_size = m_ndsize.prod();

    // allocate input and output host buffers
    auto in_A_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto out_A_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto in_B_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto out_B_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto C_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});

    // fill input and output buffers for A and B identically, C all zeros
    for (size_t i = 0; i < m_size; i++) {
        in_A_h[i] = A_intval;
        out_A_h[i] = A_intval;
        in_B_h[i] = B_intval;
        out_B_h[i] = B_intval;
        C_h[i] = 0;
    }

    //Create queue and allocate buffers on device
    auto queue = Queue{device};
    auto in_A_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto out_A_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto in_B_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto out_B_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto C_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});

    // Copy non-zero data to device
    alpaka::memcpy(queue, in_A_d, in_A_h);
    alpaka::memcpy(queue, out_A_d, out_A_h);
    alpaka::memcpy(queue, in_B_d, in_B_h);
    alpaka::memcpy(queue, out_B_d, out_B_h);

    // Fill matrix C (output) with zeros
    alpaka::memset(queue, C_d, 0x00);

    // Launch multiplication kernel
    int m_gridsize = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
    auto m_div = make_workdiv<Acc2D>({m_gridsize, m_gridsize}, {BLOCK_SIZE, BLOCK_SIZE});
    std::cout << "Testing MatMulKernel and StencilKernel with vector indices with a grid of "
              << alpaka::getWorkDiv<alpaka::Grid, alpaka::Blocks>(m_div) << " blocks x "
              << alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(m_div) << " threads x "
              << alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(m_div) << " elements...\n";
    alpaka::exec<Acc2D>(
        queue, m_div, StencilKernel{}, in_A_d.data(), out_A_d.data(), m_ndsize);
    alpaka::exec<Acc2D>(
        queue, m_div, StencilKernel{}, in_B_d.data(), out_B_d.data(), m_ndsize);
    alpaka::exec<Acc2D>(
        queue, m_div, MatMulKernel{}, out_A_d.data(), out_B_d.data(), C_d.data(), m_ndsize);

    // Copy results from device to host
    alpaka::memcpy(queue, C_h, C_d);

    // Wait for everything to finish
    alpaka::wait(queue);

    // Check results of stencils followed by matmul
    int exp_edge = A_intval*B_intval*((RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    int exp_center = A_intval*B_intval*((RADIUS*4+1)*(RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            if ((i < RADIUS || i >= DSIZE-RADIUS) && (j < RADIUS || j >= DSIZE-RADIUS)) {
                if (C_h[j+i*DSIZE] != A_intval*B_intval*DSIZE) {
                    printf("Mismatch at index [%i,%i], was: %i, should be: %i\n", i,j, C_h[j+i*DSIZE], A_intval*B_intval*DSIZE);
                    return -1;
                }
            }
            else if ((j < RADIUS || j >= DSIZE-RADIUS) && (i >= RADIUS && i< DSIZE-RADIUS)){
                if (C_h[j+i*DSIZE] != exp_edge) {
                    printf("Mismatch at index [%i,%i], was: %d, should be: %i\n", i,j, C_h[j+i*DSIZE], exp_edge);
                    return -1;
                }
            }
            else if ((i < RADIUS || i >= DSIZE-RADIUS) && (j >= RADIUS && j< DSIZE-RADIUS)){
                if (C_h[j+i*DSIZE] != exp_edge) {
                    printf("Mismatch at index [%i,%i], was: %i, should be: %i\n", i,j, C_h[j+i*DSIZE], exp_edge);
                    return -1;
                }
            }
            else {
                if (C_h[j+i*DSIZE] != exp_center) {
                    printf("Mismatch at index [%i,%i], was: %i, should be: %i\n", i,j, C_h[j+i*DSIZE], exp_center);
                    return -1;
                }
            }
        }
    }
    printf("Success!\n");
}
