import os
import numpy as np
import matplotlib.pyplot as plt

def plot_roofline(hardware_specs, kernel_data, output_path="roofline.png"):
    """
    Generates a log-log Roofline plot.
    
    :param hardware_specs: Dict containing 'max_gflops', 'bandwidth_gbps', and 'name'
    :param kernel_data: List of dicts, each with 'name', 'intensity' (FLOP/byte), and 'gflops'
    :param output_path: Path to save the resulting image
    """
    max_gflops = hardware_specs['max_gflops']
    bandwidth = hardware_specs['bandwidth_gbps']
    gpu_name = hardware_specs['name']

    # Calculate the knee point (inflection point)
    knee_point = max_gflops / bandwidth

    # Create arithmetic intensity range for the background roofline (log scale)
    # Covers typical ranges from severe memory-bound to highly compute-bound
    x_roof = np.logspace(-2, 3, 500)
    
    # Roofline equation: Attainable Performance = Min(Peak GFLOP/s, Bandwidth * Intensity)
    y_roof = np.minimum(max_gflops, bandwidth * x_roof)

    # Initialize Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Plot the Roofline Ceilings
    plt.loglog(x_roof, y_roof, color='firebrick', linewidth=2.5, label=f'{gpu_name} Theoretical Roofline')
    
    # Draw a vertical dashed line at the knee point
    plt.axvline(x=knee_point, color='gray', linestyle='--', alpha=0.7, 
                label=f'Knee Point ({knee_point:.2f} FLOP/Byte)')

    # Plot Kernel Data Points
    colors = plt.colormaps['Dark2'](np.linspace(0, 1, len(kernel_data)))
    
    for i, kernel in enumerate(kernel_data):
        ai = kernel['intensity']
        perf = kernel['gflops']
        # Calculate theoretical max at this specific intensity
        attainable = min(max_gflops, bandwidth * ai)
        efficiency = (perf / attainable) * 100

        plt.scatter(ai, perf, color=colors[i], edgecolors='k', s=100, zorder=5,
                    label=f"{kernel['name']} ({efficiency:.1f}% Opt)")
        
        # Annotate point with its name
        plt.annotate(kernel['name'], (ai, perf), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=9, weight='bold')

    # Formatting
    plt.xlabel('Arithmetic Intensity (FLOPs / Byte)', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (GFLOPs / Second)', fontsize=12, fontweight='bold')
    plt.title(f'Roofline Model Analysis ({gpu_name})', fontsize=14, fontweight='bold', pad=15)
    
    # Set reasonable axis limits based on data and specs
    plt.xlim(10**-2, 10**3)
    plt.ylim(10**0, max_gflops * 2)
    
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=10)
    plt.tight_layout()
    
    # Save and close
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Roofline plot successfully saved to: {output_path}")

if __name__ == "__main__":
    # Example Hardware Profile (e.g., NVIDIA T4 Tensor Core GPU FP32 non-tensor peak)
    # Adjust these values depending on your target architecture (T4, V100, A100, etc.)
    t4_specs = {
        'name': 'NVIDIA T4',
        'max_gflops': 8100,       # ~8.1 TFLOPs FP32 Single Precision
        'bandwidth_gbps': 320     # 320 GB/s GDDR6
    }

    # Example Kernel Benchmarking Results
    # Arithmetic Intensity (AI) = FLOPs / Total Memory Traffic (Bytes transferred to/from Global Memory)
    benchmarked_kernels = [
        {
            'name': 'Native MatMul',
            'intensity': 79.81,
            'gflops': 1.67
        },
        {
            'name': 'Tiled Vectorized MatMul',
            'intensity': 86.99,
            'gflops': 8.63
        }
    ]

    # Ensure output directory exists or handles relative workspace
    plot_roofline(t4_specs, benchmarked_kernels, output_path="benchmarks/roofline_analysis.png")
