import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Locate all generated CSV files
file_pattern = "*_rank_*.csv"
all_files = glob.glob(file_pattern)

data_records = []

# Regex to extract: implementation (custom/cublasmp), rank number, and report name
# Adjust the regex if your file prefix differs slightly (e.g., stats_custom_rank_0_...)
pattern = re.compile(r'stats_(custom|cublasmp)_rank_(\d+)_(.+)\.csv')

# 2. Parse files and extract core metrics
for file in all_files:
    match = pattern.search(file)
    if not match:
        continue
        
    implementation = match.group(1)
    rank = int(match.group(2))
    report_type = match.group(3)
    
    try:
        # Nsight CSVs sometimes have slightly varying column names, we read the raw df
        df = pd.read_csv(file)
        
        # Aggregate based on the report type
        if report_type == 'cuda_gpu_kern_sum':
            # Total GPU kernel execution time (Compute)
            time_col = [col for col in df.columns if 'Total Time' in col][0]
            total_time_ms = df[time_col].sum() / 1e6 # Convert ns to ms
            data_records.append({'Implementation': implementation, 'Rank': rank, 'Metric': 'Compute_Time_ms', 'Value': total_time_ms})
            
        elif report_type == 'mpi_event_sum':
            # Total MPI communication time (Overhead)
            time_col = [col for col in df.columns if 'Total Time' in col][0]
            total_time_ms = df[time_col].sum() / 1e6 # Convert ns to ms
            data_records.append({'Implementation': implementation, 'Rank': rank, 'Metric': 'MPI_Time_ms', 'Value': total_time_ms})
            
        elif report_type == 'cuda_gpu_mem_size_sum':
            # Total memory transfer size
            size_col = [col for col in df.columns if 'Total' in col and 'MB' in col][0]
            total_mb = df[size_col].sum()
            data_records.append({'Implementation': implementation, 'Rank': rank, 'Metric': 'Mem_Transfer_MB', 'Value': total_mb})

    except Exception as e:
        print(f"Skipping {file} due to parsing error: {e}")

# 3. Create a clean analytical dataframe
df_results = pd.DataFrame(data_records)

if df_results.empty:
    print("No matching data found. Please verify the CSV file names in your directory.")
else:
    # Pivot the table for a clean side-by-side view
    summary_table = df_results.pivot_table(
        index=['Implementation', 'Rank'], 
        columns='Metric', 
        values='Value', 
        aggfunc='sum'
    ).reset_index()

    print("=== 1:1 Profiling Summary ===")
    print(summary_table.to_string(index=False))

    # 4. Visualize Compute vs. Communication Overlap
    # Filter for the timing metrics
    plot_data = df_results[df_results['Metric'].isin(['Compute_Time_ms', 'MPI_Time_ms'])]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_data, 
        x='Rank', 
        y='Value', 
        hue=plot_data[['Implementation', 'Metric']].apply(tuple, axis=1),
        palette='viridis'
    )
    plt.title('Compute vs. MPI Communication Overhead by Rank')
    plt.ylabel('Total Time (ms)')
    plt.xlabel('MPI Rank')
    plt.legend(title='Implementation & Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
