import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_nsight_csv_report(file_path):
    """
    Robustly parses text-aligned or space-padded Nsight Compute report tables.
    Removes C++ template decorations and aggregates metrics across all profiling passes.
    """
    if not os.path.exists(file_path):
        print(f"[WARNING] Profile file not found at: {file_path}")
        return {}

    kernel_profiles = {}
    current_kernel = None

    with open(file_path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            # Detect kernel declaration lines containing execution configurations (e.g. Context, Stream, Device)
            if "(" in line_str and any(kw in line_str for kw in ["Context", "Stream", "Device", "CC"]):
                # Isolate function portion before parameters
                func_part = line_str.split('(')[0].strip()
                # Strip out C++ template parameters (<...>) to get base name
                if '<' in func_part:
                    func_part = func_part.split('<')[0].strip()
                # Split by space or scope resolutions to isolate identifier
                parts = re.split(r'\s+|::', func_part)
                current_kernel = parts[-1].strip()
                
                if current_kernel not in kernel_profiles:
                    kernel_profiles[current_kernel] = {'bytes': 0.0, 'fadd': 0, 'fmul': 0, 'ffma': 0}
                continue

            if not current_kernel:
                continue

            # Match hardware metric rows
            if any(m in line_str for m in ["dram__bytes.sum", "fadd_pred_on.sum", "fmul_pred_on.sum", "ffma_pred_on.sum"]):
                parts = re.split(r'\s{2,}', line_str)
                if len(parts) >= 2:
                    metric_name = parts[0].strip()
                    metric_val_str = parts[-1].strip().replace(',', '')

                    try:
                        value = float(metric_val_str)
                    except ValueError:
                        continue

                    if "dram__bytes.sum" == metric_name:
                        if "Mbyte" in line_str:
                            kernel_profiles[current_kernel]['bytes'] += value * 1_000_000
                        elif "Kbyte" in line_str:
                            kernel_profiles[current_kernel]['bytes'] += value * 1_000
                        else:
                            kernel_profiles[current_kernel]['bytes'] += value
                    elif "fadd_pred_on.sum" in metric_name:
                        kernel_profiles[current_kernel]['fadd'] += int(value)
                    elif "fmul_pred_on.sum" in metric_name:
                        kernel_profiles[current_kernel]['fmul'] += int(value)
                    elif "ffma_pred_on.sum" in metric_name:
                        kernel_profiles[current_kernel]['ffma'] += int(value)

    # Compute final dynamic Arithmetic Intensity mappings
    calculated_ai_coords = {}
    for name, stats in kernel_profiles.items():
        total_flops = stats['fadd'] + stats['fmul'] + (stats['ffma'] * 2)
        total_bytes = stats['bytes']
        ai = total_flops / total_bytes if total_bytes > 0 else 0.0
        calculated_ai_coords[name] = ai
        print(f"[PARSED] Kernel: {name:<35} | Active AI: {ai:.4f} FLOP/B")

    return calculated_ai_coords


# --- Hardware Specs (NVIDIA T4 Architecture) ---
t4_specs = {
    'name': 'NVIDIA T4',
    'max_gflops': 8100,       # Single-Precision Peak GFLOPs/s
    'bandwidth_gbps': 320     # Peak Memory Bandwidth GB/s
}

# --- Profile Report Ingestion Paths ---
# Adjust file paths here if your directory layout differs
matmul_file = "benchmarks/matMulVecMem_report.csv"
cholesky_file = "benchmarks/cholesky_tiled_report.csv"
cg_file = "benchmarks/conjugate_gradient_2d_report.csv"
cusparse_spmv_file = "benchmarks/cuSPARSE_spmv_csr_report.csv"
spmv_csr_file = "benchmarks/spmv_csr_report.csv"

matmul_ai = parse_nsight_csv_report(matmul_file)
cholesky_ai = parse_nsight_csv_report(cholesky_file)
cg_ai = parse_nsight_csv_report(cg_file)
cusparse_ai = parse_nsight_csv_report(cusparse_spmv_file)
spmv_ai = parse_nsight_csv_report(spmv_csr_file)

# --- Consolidated Kernel Benchmark Matrix ---
benchmarked_kernels = [
    # 1. Dense GEMM Domain (Circles 'o')
    {
        'name': 'Native MatMul',
        'intensity': matmul_ai.get("multiplyKernelNative", 80.00),
        'gflops': 1.67,
        'marker': 'o'
    },
    {
        'name': 'Tiled Vectorized MatMul',
        'intensity': matmul_ai.get("multiplyKernelTiledVectorizedA", 86.56),
        'gflops': 8.63,
        'marker': 'o'
    },
    
    # 2. Dense Matrix Solvers Domain (Squares 's')
    {
        'name': 'Cholesky Diagonal',
        'intensity': cholesky_ai.get("diagnoal_factorization_kernel", 0.8121),
        'gflops': 1.5,
        'marker': 's'
    },
    {
        'name': 'Cholesky Column Update',
        'intensity': cholesky_ai.get("column_update_kernel", 5.9258),
        'gflops': 42.0,
        'marker': 's'
    },
    {
        'name': 'Cholesky Trailing Submatrix',
        'intensity': cholesky_ai.get("trailing_submatrix_kernel", 6.6836),
        'gflops': 135.2,
        'marker': 's'
    },

    # 3. Sparse Kernels Domain (Triangles '^')
    {
        'name': 'Custom CSR SpMV (Scalar)',
        'intensity': spmv_ai.get("spmv_csr_scalar_kernel", 0.1015),
        'gflops': 0.0145,
        'marker': '^'
    },
    {
        'name': 'Custom CSR SpMV (Vector)',
        'intensity': spmv_ai.get("spmv_csr_vector_kernel", 0.2521),
        'gflops': 0.0419,
        'marker': '^'
    },
    {
        'name': 'cuSPARSE CSR SpMV',
        'intensity': cusparse_ai.get("csrmv_v3_kernel", 0.2884),
        'gflops': 0.0397,
        'marker': 'v'
    },

    # 4. Iterative Solvers / Krylov Subspace Domain (Diamonds 'D')
    {
        'name': 'CG SpMV Bottleneck',
        'intensity': cg_ai.get("spmv_csr_vector_kernel", 1.5813),
        'gflops': 0.0661,
        'marker': 'D'
    },
    {
        'name': 'CG Vector Update X/R',
        'intensity': cg_ai.get("update_x_r_kernel", 0.1742),
        'gflops': 0.0419,
        'marker': 'D'
    },
    {
        'name': 'CG Vector Update P',
        'intensity': cg_ai.get("update_p_kernel", 0.1942),
        'gflops': 0.0386,
        'marker': 'D'
    }
]

# --- Render the Log-Log Roofline Space ---
max_gflops = t4_specs['max_gflops']
bandwidth = t4_specs['bandwidth_gbps']
knee_point = max_gflops / bandwidth

x_roof = np.logspace(-2, 3, 500)
y_roof = np.minimum(max_gflops, bandwidth * x_roof)

plt.figure(figsize=(11, 7), dpi=150)
plt.grid(True, which="both", ls="-", alpha=0.3)

# Draw rooflines
plt.loglog(x_roof, y_roof, color='firebrick', linewidth=2.5, label="Theoretical Peak Ceiling")
plt.axvline(x=knee_point, color='gray', linestyle='--', alpha=0.6, label=f"Knee Point ({knee_point:.2f} FLOP/B)")

# Generate unique colors across benchmarks
colors = plt.cm.tab10(np.linspace(0, 1, len(benchmarked_kernels)))

for i, kernel in enumerate(benchmarked_kernels):
    ai = kernel['intensity']
    perf = kernel['gflops']
    attainable = min(max_gflops, bandwidth * ai)
    efficiency = (perf / attainable) * 100

    plt.scatter(ai, perf, color=colors[i], marker=kernel['marker'], edgecolors='k', s=120, zorder=5,
                label=f"{kernel['name']} ({efficiency:.2f}% Opt)")
    
    # --- CHANGE THIS SECTION TO INDIVIDUALLY OFFSET LABELS ---
    # Default offsets
    x_offset = 6
    y_offset = 8 if i % 2 == 0 else -14
    
    # Override specifically for cuSPARSE to push it away from the custom SpMV dots
    if kernel['name'] == 'cuSPARSE CSR SpMV':
        x_offset = 10
        y_offset = -22  # Pushes the text further down out of the way
    elif kernel['name'] == 'Custom CSR SpMV (Vector)':
        x_offset = -10
        y_offset = 12   # Shifts this slightly left and up
        
    plt.annotate(kernel['name'], (ai, perf), textcoords="offset points", 
                 xytext=(x_offset, y_offset), ha='left', va='center', fontsize=8, weight='bold', alpha=0.8)
plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=11, fontweight='bold')
plt.ylabel('Performance (GFLOPs / Second)', fontsize=11, fontweight='bold')
plt.title(f"HPC Suite Performance Roofline Analysis ({t4_specs['name']})", fontsize=13, fontweight='bold', pad=12)
plt.xlim(10**-2, 10**3)
plt.ylim(10**-3, max_gflops * 2)

plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, edgecolor='gray', fontsize=8.5)
plt.tight_layout()

# Save image file back to repository artifacts
output_img = "benchmarks/roofline_analysis.png"
os.makedirs(os.path.dirname(output_img), exist_ok=True)
plt.savefig(output_img, bbox_inches='tight')
plt.close()

print(f"[SUCCESS] Roofline map with {len(benchmarked_kernels)} data streams saved to: {output_img}")
