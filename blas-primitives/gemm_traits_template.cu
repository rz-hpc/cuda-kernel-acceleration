
#include <cuda_runtime.h>
#include <iostream>
#include <cuda_fp16.h>
#include <type_traits>
#include <limits>

// Layer 1: gemm traits struct

// Traits are compile-time lookup tables keyed on a type,
// so need full tempalte specialization for later "if constexpr"
// to brach within a function body when T is known.
// Specialization picks a variant, if constexpr prunes dead branches

// Primary template -- default
template <typename T>
struct gemm_traits {
    using compute_t = T; // type used for accumulation
    using scalar_t = T;  // type used for alpha/beta
    static constexpr float tolerance = 1e-4f;
    static constexpr bool use_tensor_cores = false;
    static constexpr const char* name = "fp32_fallback";
};

// Specialization for half precision
// Compute/Accumulate in half precision overflows/loses precision fast
// so tensor-core kernels always accumulate in fp32 even when i/o are half
template <>
struct gemm_traits<half> {
  using compute_t = float; // accumulate in fp32 even though inputs are half
  using scalar_t = half;
  static constexpr float tolerance = 1e-2f; // looser -- half has ~3 decimal digits
  static constexpr bool use_tensor_cores = true;
  static constexpr const char* name = "fp16_tensorcore";
};

// Specialization for double
template <>
struct gemm_traits<double> {
    using compute_t = double;
    using scalar_t = double;
    static constexpr float tolerence = 1e-8f;
    static constexpr bool use_tensor_cores = false;
    static constexpr const char* name = "fp64_fallback";
};

// Layer 2: if constexpr dispatch inside the kernel launcher

template <typename T>
void gemm_dispatch(const T* A, const T* B, T* C,
                    int M, int N, int K,
                    typename gemm_traits<T>::scalar_t alpha,
                    typename gemm_traits<T>::scalar_t beta) {

      using traits = gemm_traits<T>;

      if constexpr (traits::use_tensor_cores) {
          // only compiles/instantiates for T = half
          // WMMA API calls (nvcuda::wmma::fragment, etc)
          launch_gemm_tensorcore<T, typename traits::compute_t>(A, B, C, M, N. K, alpha, beta);
      }
      else {
          // tiled FP32/FP64 kernel
          launch_gemm_tiled<T>(A, B, C, M, N, K, alpha, beta);
      }
}

// Layer 3: tolerance-based verification

template <typename T>
bool verify_gemm(const T* result, const T* reference, int size) {
    using traits = gemm_traits<T>;
    float max_rel_error = 0.0f;

    for (int i = 0; i < size; i++) {
        float ref = static_cast<float>(reference[i]);
        float res = static_cast<float>(result[i]);
        float rel_error = std::abs(ref - res) / (std::abs(ref) + 1e-8f);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    printf("[%s] max relative error: %e (tolerance: %e)\n",
          traits::name, max_rel_error, traits::tolerance);

    return max_rel_error < traits::tolerance;
}

