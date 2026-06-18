import os
import re

def extract_gflops_from_csv(file_path):
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found at: {file_path}")
        return
    
    current_kernel = None
    stats = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Detect unique kernel configuration headers
            if "(" in line_str and any(kw in line_str for kw in ["Context", "Stream", "Device", "CC"]):
                func_part = line_str.split('(')[0].strip()
                if '<' in func_part:
                    func_part = func_part.split('<')[0].strip()
                parts = re.split(r'\s+|::', func_part)
                current_kernel = parts[-1].strip()
                
                if current_kernel not in stats:
                    stats[current_kernel] = {'fadd': 0, 'fmul': 0, 'ffma': 0, 'duration_msec': 0.0}
                continue
            
            if not current_kernel:
                continue
                
            # Accumulate raw execution counters and duration values
            if any(m in line_str for m in ["fadd_pred_on.sum", "fmul_pred_on.sum", "ffma_pred_on.sum", "gpu__time_duration.sum"]):
                parts = re.split(r'\s{2,}', line_str)
                if len(parts) >= 2:
                    metric_name = parts[0].strip()
                    val_str = parts[-1].strip().replace(',', '')
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                    
                    if "fadd_pred_on.sum" in metric_name:
                        stats[current_kernel]['fadd'] += int(val)
                    elif "fmul_pred_on.sum" in metric_name:
                        stats[current_kernel]['fmul'] += int(val)
                    elif "ffma_pred_on.sum" in metric_name:
                        stats[current_kernel]['ffma'] += int(val)
                    elif "gpu__time_duration.sum" in metric_name:
                        # CRITICAL FIX: Match explicit sub-strings before checking generic 'second' fallback
                        if "msecond" in line_str:
                            stats[current_kernel]['duration_msec'] += val
                        elif "usecond" in line_str:
                            stats[current_kernel]['duration_msec'] += val / 1000.0
                        elif "nsecond" in line_str:
                            stats[current_kernel]['duration_msec'] += val / 1000000.0
                        elif "second" in line_str:
                            stats[current_kernel]['duration_msec'] += val * 1000.0
                        else:
                            # If no unit token is present, assume raw ms from Nsight default configuration
                            stats[current_kernel]['duration_msec'] += val

    print(f"\n=== Performance Extracted From: {os.path.basename(file_path)} ===")
    for name, data in stats.items():
        total_flops = data['fadd'] + data['fmul'] + (data['ffma'] * 2)
        
        # GFLOPs/sec = (Total FLOPs / 1e9) / (Duration_msec / 1e3)
        # Simplified: Total FLOPs / (Duration_msec * 1,000,000.0)
        if data['duration_msec'] > 0 and total_flops > 0:
            gflops_per_sec = total_flops / (data['duration_msec'] * 1000000.0)
            print(f"  Kernel: {name:<32} | Time: {data['duration_msec']:11.3f} ms | Achieved: {gflops_per_sec:10.4f} GFLOPs/s")
        elif total_flops == 0:
            print(f"  Kernel: {name:<32} | Time: {data['duration_msec']:11.3f} ms | Pure Memory/Setup (0 FLOPs)")

# Execute clean parses across your local suite directory
extract_gflops_from_csv("benchmarks/cuSPARSE_spmv_csr_report.csv")
extract_gflops_from_csv("benchmarks/spmv_csr_report.csv")
extract_gflops_from_csv("benchmarks/conjugate_gradient_2d_report.csv")
extract_gflops_from_csv("benchmarks/matMulVecMem_report.csv")
extract_gflops_from_csv("benchmarks/cholesky_tiled_report.csv")
extract_gflops_from_csv("benchmarks/cuDSS_Refactorization_report.csv")
