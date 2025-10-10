#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define BLOCK_DIM 32
#define MAT_DIM 1024

// Basically just a checkCuda to see if there was an error
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const* const func, const char* const file, int const line)
{
  if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Create a vector of size n with random values between -256 and 256
template <typename T>
std::vector<T> create_rand_vector(size_t n)
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<int> uniform_dist(-256, 256);
    std::vector<T> vec(n);
    for (size_t i = 0; i < n; ++i)
    {
        vec.at(i) = static_cast<T>(uniform_dist(e));
    }
    return vec;
}

// mat_1 is m x n
// mat_2 is n x p
// mat_3 is m x p
template <typename T>
void mm(T const* mat_1, T const* mat_2, T* mat_3, size_t m, size_t n, size_t p)
{
    // Compute Sequentially
    for (size_t i{0}; i < m; ++i)
    {
        for (size_t j{0}; j < p; ++j)
        {
            T sum{0};
            for (size_t k{0}; k < n; ++k)
            {
                sum += mat_1[i * n + k] * mat_2[k * p + j];
            }
            mat_3[i * p + j] = sum;
        }
    }
}

template <typename T>
__global__ void mm_kernel(T const* mat_1, T const* mat_2, T* mat_3, size_t m, size_t n, size_t p)
{
    // 2D block and 2D thread
    size_t i{blockIdx.x * blockDim.x + threadIdx.x};
    size_t j{blockIdx.y * blockDim.y + threadIdx.y};

    // Bound prevention
    if ((i >= m)) || (j >= p)
    {
        return;
    }

    // Just do the one column and row pair for this thread
    T sum{0};
    for (size_t k{0}; k < n; ++k)
    {
        sum += mat_1[i * n + k] * mat_2[k * p + j];
    }
    mat_3[i * p + j] = sum;
}

template <typename T>
void mm_cuda(T const* mat_1, T const* mat_2, T* mat_3, size_t m, size_t n, size_t p)
{
    dim3 block_per_grid(1,1);
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);

    blocks_per_grid.x = std::ceil(static_cast<double>(p) / static_cast<double>(threads_per_block.x));
    blocks_per_grid.y = std::ceil(static_cast<double>(m) / static_cast<double>(threads_per_block.y));

    mm_kernel<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n, p);
    
    // Memory transfer so auto synchronization happens
    //checkCuda(cudaDeviceSynchronize());
}

// Check if two vectors are element-wise equal within a tolerance
template <typename T>
bool allclose(std::vector<T> const& vec_1, std::vector<T> const& vec_2, T const& abs_tol){
    if (vec_1.size() != vec_2.size())
    {
        return false;
    }
    for (size_t i{0}; i < vec_1.size(); ++i)
    {
        if (std::abs(vec_1.at(i) - vec_2.at(i)) > abs_tol)
        {
            std::cout << vec_1.at(i) << " " << vec_2.at(i) << std::endl;
            return false;
        }
    }
    return true;
}

// Test function for random matrices of size m x n and n x p
template <typename T>
bool random_test_mm_cuda(size_t m, size_t n, size_t p)
{
    std::vector<T> const mat_1_vec{create_rand_vector<T>(m * n)};
    std::vector<T> const mat_2_vec{create_rand_vector<T>(n * p)};
    std::vector<T> mat_3_vec(m * p);
    std::vector<T> mat_4_vec(m * p);
    T const* mat_1{mat_1_vec.data()};
    T const* mat_2{mat_2_vec.data()};
    T* mat_3{mat_3_vec.data()};
    T* mat_4{mat_4_vec.data()};

    mm(mat_1, mat_2, mat_3, m, n, p);

    T *d_mat_1, *d_mat_2, *d_mat_4;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * mat_1_vec.size()));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * mat_2_vec.size()));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(T) * mat_4_vec.size()));

    // Copy data from host to device.
    checkCuda(cudaMemcpy(d_mat_1, mat_1, sizeof(T) * mat_1_vec.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_mat_2, mat_2, sizeof(T) * mat_2_vec.size(), cudaMemcpyHostToDevice));

    // Run matrix multiplication on GPU.
    mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
    cudaDeviceSynchronize();
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute." << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Copy data from device to host.
    checkCuda(cudaMemcpy(mat_4, d_mat_4, sizeof(T) * mat_4_vec.size(), cudaMemcpyDeviceToHost));

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_4));

    return allclose<T>(mat_3_vec, mat_4_vec, 1e-4);
}


// Run multiple tests of random matrices of size m x n and n x p
template <typename T>
bool random_multiple_test_mm_cuda(size_t num_tests)
{
    size_t m{MAT_DIM}, n{MAT_DIM}, p{MAT_DIM};
    bool success{false};
    for (size_t i{0}; i < num_tests; ++i)
    {
        success = random_test_mm_cuda<T>(m, n, p);
        if (!success)
        {
            return false;
        }
    }

    return true;
}


// Measure latency of matrix multiplication on GPU
template <typename T>
float measure_latency_mm_cuda(size_t m, size_t n, size_t p, size_t num_tests, size_t num_warmups)
{
    cudaEvent_t startEvent, stopEvent;
    float time{0.0f};

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    T *d_mat_1, *d_mat_2, *d_mat_4;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * m * n));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * n * p));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(T) * m * p));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
    }

    checkCuda(cudaEventRecord(startEvent, 0));
    for (size_t i{0}; i < num_tests; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute." << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_4));

    float latency{time / num_tests};

    return latency;
}

// Host Driver
int main()
{
    const size_t num_tests{2};
    assert(random_multiple_test_mm_cuda<int32_t>(num_tests));
    assert(random_multiple_test_mm_cuda<float>(num_tests));
    assert(random_multiple_test_mm_cuda<double>(num_tests));
    std::cout << "All tests passed!\n";

    // Latency measurement parameters
    const size_t num_measurement_tests{2};
    const size_t num_measurement_warmups{1};
    size_t m{MAT_DIM}, n{MAT_DIM}, p{MAT_DIM};

    // Measure latency for int32, float, and double
    float mm_cuda_int32_latency = measure_latency_mm_cuda<int32_t>(m, n, p, num_measurement_tests, num_measurement_warmups);
    float mm_cuda_float_latency = measure_latency_mm_cuda<float>(m, n, p, num_measurement_tests, num_measurement_warmups);
    float mm_cuda_double_latency = measure_latency_mm_cuda<double>(m, n, p, num_measurement_tests, num_measurement_warmups);

    // Print results
    std::cout << " Matrix Multiplication Runtime \n";
    std::cout << "m: " << m << " n: " << n << " p: " << p << "\n";
    std::cout << " INT32 : " << mm_cuda_int32_latency << " ms\n";
    std::cout << " FLOAT : " << mm_cuda_float_latency << " ms\n";
    std::cout << " DOUBLE : " << mm_cuda_double_latency << " ms\n";
    return 0;
}