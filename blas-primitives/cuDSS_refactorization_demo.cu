
// Demo: repeated solves on matrics with a shared sparsity pattern
// Scenario: solve A_k x_k = b_k, for k = 0, 1, ...NUM_SYSTEM - 1
// where every A_K has the same non-zero structure (rowPtr and colIndices)
// but different numerical values.

// * Matrix used:
// *   A 5×5 SPD tridiagonal system (easy to verify analytically):
// *
// *     [ d  -1        ]
// *     [-1   d  -1    ]
// *     [    -1   d  -1]
// *     [        -1  d ]   — scaled by a per-system factor α_k
// *
// *   x_k = A_k^{-1} b_k,  b_k = ones vector

//cuDSS
// Initialization: handle -- run once (per sparsity pattern)
// Analysis (on CPU): reordering, symbolic factorization -- run once
// Factorization/Refactorization -- once per k
// Solve: ly = pb, ux = y -- once per k (or per RHS)
// Clean up -- run once

/* To compile and run on Google Colab
# 1. Manually set the filename
!rm -rf libcudss-*
!wget -q https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda12-archive.tar.xz

# 2. Extract explicitly
!tar -xf libcudss-linux-x86_64-0.8.0.10_cuda12-archive.tar.xz

# 3. Compile using explicit hardcoded paths
!nvcc cuDSS_refactorization_demo.cu \
    -I/content/libcudss-linux-x86_64-0.8.0.10_cuda12-archive/include \
    -L/content/libcudss-linux-x86_64-0.8.0.10_cuda12-archive/lib \
    -lcudss -lcusolver -lcublas \
    -Xlinker -rpath=/content/libcudss-linux-x86_64-0.8.0.10_cuda12-archive/lib \
    -o cuDSS_refactorization_demo

# 4. Run
!./cuDSS_refactorization_demo
*/

#include <cuda_runtime.h>
#include <cudss.h>

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(call) {\
  cudaError_t err = call;\
  if (err != cudaSuccess) {\
    printf("Error: %s in %s at %d!", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}

#define CHECK_CUDSS(call) {\
  cudssStatus_t status = call;\
  if (status != CUDSS_STATUS_SUCCESS) {\
    printf("cuDSS Error: %d in %s at %d!", status, __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}

static const int N = 5; // matrix dimesion
static const int NUM_SYSTEMS = 4; // number of repeated solves
static const double DIAG_BASE = 4.0; // diagonal value for alpha = 1

// Build CSR arrays for the tridiagonal matrix scaled by alpha
// A = alpha * tridiag(-1, DIAG_BASE, -1)
// CSR with 0-based indexing
static void build_tridiagonal_csr(int n, double alpha,
                                  std::vector<int>& row_ptr,
                                  std::vector<int>& col_indices,
                                  std::vector<double>& values){
  row_ptr.clear(); col_indices.clear(); values.clear();
  row_ptr.resize(n + 1, 0);

  for (int i = 0; i < n; i++) {
    int nnz_row = 1; // diagonal non-zero
    if (i > 0) ++nnz_row; // sub-diagonal
    if (i < n - 1) ++nnz_row; // super-diagonal
    row_ptr[i + 1] = row_ptr[i] + nnz_row;
  }

  int nnz = row_ptr[n];
  col_indices.resize(nnz);
  values.resize(nnz);

  int idx = 0;
  for (int i = 0; i < n; i++) {
      if (i > 0) {
        col_indices[idx] = i - 1;
        values[idx] = alpha * (-1.0);
        ++idx;
      }
      col_indices[idx] = i;
      values[idx] = alpha * DIAG_BASE;
      ++idx;
      if (i < n - 1) {
        col_indices[idx] = i + 1;
        values[idx] = alpha * (-1.0);
        ++idx;
      }
  }
}

// Helper reference solve via Guassian elemination
static void reference_solve_guassian(int n , double alpha,
                            const double *b, double *x_ref) {
  // Build dense matrix
  std::vector<double> A(n*n, 0.0);
  for (int i = 0; i < n; i++) {
    A[i * n + i] = alpha * DIAG_BASE;
    if (i > 0) A[i * n + (i - 1)] = alpha * (-1.0);
    if (i < n - 1) A[i * n + (i + 1)] = alpha * (-1.0);
  }
  std::vector<double> rhs(b, b + n);

  // Gaussian elemination with partial pivoting
  for (int col = 0; col < n; col++) {
    int pivot = col;
    for (int r = col + 1; r < n; r++) {
      if (fabs(A[r * n + col]) > fabs(A[pivot * n + col]))
        pivot = r;
      if (pivot != col) {
        for (int c = 0; c < n; c++) {
          std::swap(A[col * n + c], A[pivot * n + c]);
        }
        std::swap(rhs[col], rhs[pivot]);
      }
      for (int r = col + 1; r < n; r++) {
        double f = A[r * n + col] / A[col * n + col];
        for (int c = col; c < n; c++) {
          A[r * n + c] -= f * A[col * n + c];
        }
        rhs[r] -= f * rhs[col];
      }
    }
    for (int i = n - 1; i >= 0; i--) {
      x_ref[i] = rhs[i];
      for (int j = i + 1; j < n; j++) {
        x_ref[i] -= A[i * n + j] * x_ref[j];
      }
      x_ref[i] /= A[i * n + i];
    }
  }
}

int main() {

  std::vector<int> h_row_ptr, h_col_indices;
  std::vector<double> h_values;
  build_tridiagonal_csr(N, 1.0, h_row_ptr, h_col_indices, h_values);
  int nnz = (int)h_values.size();

  std::vector<double> h_b(N, 1.0);
  std::vector<double> h_x(N, 0.0);
  std::vector<double> x_ref(N, 0.0);

  int *d_row_ptr, *d_col_indices;
  double *d_values, *d_b, *d_x;

  CHECK_CUDA(cudaMalloc(&d_row_ptr, (N + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_values, nnz * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(double)));
  CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(double)));

  CHECK_CUDA(cudaMemcpy(d_row_ptr, h_row_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));

  // 1. Initilization (handle, config, data) -- run once
  cudssHandle_t handle;
  cudssConfig_t config;
  cudssData_t data;

  CHECK_CUDSS(cudssCreate(&handle));
  CHECK_CUDSS(cudssConfigCreate(&config));
  CHECK_CUDSS(cudssDataCreate(handle, &data));

  // 2. Create matrix descriptors (no data copy)
  cudssMatrix_t matA, matB, matX;

  CHECK_CUDSS(cudssMatrixCreateCsr(
    &matA,
    N, N, nnz,
    d_row_ptr, nullptr,      // rowStart, rowEnd
    d_col_indices, d_values,
    CUDSS_R_32I,               // offsetType  (type of rowStart/rowEnd = int)
    CUDSS_R_32I,               // indexType   (type of colIndices = int)
    CUDSS_R_64F,          // valueType
    CUDSS_MTYPE_GENERAL,     // mtype
    CUDSS_MVIEW_FULL,        // mview
    CUDSS_BASE_ZERO));       // indexBase  ← now last, not in the middle

  // Dense RHS b: Nx1 column-major
  CHECK_CUDSS(cudssMatrixCreateDn(
    &matB, N, 1, N, d_b, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  
  // Dense solution x: Nx1 column-major
  CHECK_CUDSS(cudssMatrixCreateDn(
    &matX, N, 1, N, d_x, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));

  // 3. Anaylsis -- run once for a shared sparsity pattern
  CHECK_CUDSS(cudssExecute(
    handle,
    CUDSS_PHASE_ANALYSIS,
    config,
    data,
    matA,
    matX,
    matB));

  for (int k = 0; k < NUM_SYSTEMS; k++) {
    double alpha = 1.0 + 0.5 * k; // A_k = alpha_k * A_0

    // update numerical values on device
    build_tridiagonal_csr(N, alpha, h_row_ptr, h_col_indices, h_values);
    CHECK_CUDA(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDSS(cudssMatrixSetValues(matA, d_values));

    // 4. (Re)Factorization-- once per k
    // Factorization (first system) or refactorization (subsequent)
    cudssPhase_t factor_phase = (k == 0) ? CUDSS_PHASE_FACTORIZATION : CUDSS_PHASE_REFACTORIZATION;

    CHECK_CUDSS(cudssExecute(
      handle,
      factor_phase,
      config,
      data,
      matA,
      matX,
      matB));

      // 5. Solve -- once every RHS b
      // triangular solves + optional iterative refinement
      // ly = pb, ux = y
      CHECK_CUDSS(cudssExecute(
        handle,
        CUDSS_PHASE_SOLVE,
        config,
        data,
        matA,
        matX,
        matB));

      // Copy solution back to host and verify
      CHECK_CUDA(cudaMemcpy(h_x.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

      reference_solve_guassian(N, alpha, h_b.data(), x_ref.data());

      double max_err = 0.0;
      for (int i = 0; i < N; i++) {
        max_err = fmax(max_err, fabs(h_x[i] - x_ref[i]));
      }

      if (max_err > 1e-4)
        std::cout << "Solve Failed: System " << k << " max error " << max_err << "\n";
      else
        std::cout << "System " << k << " (alpha=" << alpha << "): OK, max_err=" << max_err << "\n";     
  }

  // 6. Clean up--reverse order
  CHECK_CUDSS(cudssMatrixDestroy(matA));
  CHECK_CUDSS(cudssMatrixDestroy(matB));
  CHECK_CUDSS(cudssMatrixDestroy(matX));
  CHECK_CUDSS(cudssDataDestroy(handle, data));
  CHECK_CUDSS(cudssConfigDestroy(config));
  CHECK_CUDSS(cudssDestroy(handle));

  CHECK_CUDA(cudaFree(d_row_ptr));
  CHECK_CUDA(cudaFree(d_col_indices));
  CHECK_CUDA(cudaFree(d_values));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_x));

  return 0;
}
