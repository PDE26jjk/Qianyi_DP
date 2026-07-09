#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <vector>
#include <random>
#include <string>
#include <iostream>
#include "benchmarks.h"


static std::vector<unsigned int> generateRandomData(size_t N, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<unsigned int> dist(0, std::numeric_limits<unsigned int>::max());
    std::vector<unsigned int> data(N);
    for (auto& v : data) v = dist(gen);
    return data;
}

void sort_benchmark(const std::vector<size_t>& sizes, int warmup, int runs,bool verify) {
    auto& timer = globalTimer();

    for (size_t N : sizes) {
        // 生成随机数据
        auto host_data = generateRandomData(N);
        if (N <= 100000) {
            std::string cpu_name = "std_sort_" + std::to_string(N);
            // 多次运行取平均
            for (int i = 0; i < runs; ++i) {
                std::vector<unsigned int> cpu_copy = host_data;
                timer.start(cpu_name);
                std::sort(cpu_copy.begin(), cpu_copy.end());
                timer.stop();
            }
        }
        // 分配 GPU 内存
        unsigned int *d_original, *d_thrust_work;
        unsigned int *d_cub_input, *d_cub_output;
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        cudaMalloc(&d_original, N * sizeof(unsigned int));
        cudaMalloc(&d_thrust_work, N * sizeof(unsigned int));
        cudaMalloc(&d_cub_input, N * sizeof(unsigned int));
        cudaMalloc(&d_cub_output, N * sizeof(unsigned int));
        cudaMemcpy(d_original, host_data.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // 计算 CUB 所需临时存储
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_cub_input, d_cub_output, N);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // 预热（不计时）
        for (int i = 0; i < warmup; ++i) {
            cudaMemcpy(d_thrust_work, d_original, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            thrust::sort(thrust::device, d_thrust_work, d_thrust_work + N);
            cudaMemcpy(d_cub_input, d_original, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_cub_input, d_cub_output, N);
        }

        cudaDeviceSynchronize();

        // 带规模信息的计时名称
        std::string thrust_name = "thrust_" + std::to_string(N);
        std::string cub_name    = "cub_"    + std::to_string(N);

        // 正式计时 runs 次
        for (int i = 0; i < runs; ++i) {
            // Thrust 排序
            cudaMemcpy(d_thrust_work, d_original, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            timer.start(thrust_name);
            thrust::sort(thrust::device, d_thrust_work, d_thrust_work + N);
            cudaDeviceSynchronize();
            timer.stop();

            // CUB 排序
            cudaMemcpy(d_cub_input, d_original, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            timer.start(cub_name);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_cub_input, d_cub_output, N);
            cudaDeviceSynchronize();
            timer.stop();
        }
        if (verify) {
            // 将最后一次 CUB 结果拷回 Host
            std::vector<unsigned int> result(N);
            cudaMemcpy(result.data(), d_cub_output, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            bool sorted = std::is_sorted(result.begin(), result.end());
            // 检查元素和是否相等（避免重复元素导致的错误）
            unsigned long long sum_orig = 0, sum_sorted = 0;
            for (auto v : host_data) sum_orig += v;
            for (auto v : result) sum_sorted += v;

            if (sorted && sum_orig == sum_sorted) {
                std::cout << "[PASS] Size " << N << " : CUB sort correct." << std::endl;
            } else {
                std::cerr << "[FAIL] Size " << N << " : sorted=" << sorted
                    << ", sum_orig=" << sum_orig << ", sum_sorted=" << sum_sorted << std::endl;
            }
        }

        // 清理
        cudaFree(d_original);
        cudaFree(d_thrust_work);
        cudaFree(d_cub_input);
        cudaFree(d_cub_output);
        cudaFree(d_temp_storage);
    }
}
