import json
import os
import numpy as np
import subprocess
import argparse
import matplotlib.pyplot as plt

def draw_and_save_plot(x, y_series_dict, x_label, y_label, plot_title, file_name):
    plt.figure(figsize=(10, 6))
    
    colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['s', 'o', '^', 'v', 'd', 'x', '+', '*']
    
    for i, (series_name, y_data) in enumerate(y_series_dict.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(x, y_data, marker=marker, color=color, label=series_name, linewidth=2, markersize=6)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"plot/{file_name}", dpi=300, bbox_inches='tight')
    plt.close()

def check_correctness(A, B, C):
    eps = np.finfo(np.float64).eps
    n = A.shape[0]
    assert np.linalg.norm(C - A @ B) < eps * n * np.linalg.norm(A) * np.linalg.norm(B), "Matrix multiplication result is incorrect!"

def run_dgemm(kernel, A, B, num_threads=1, max_time=1, trials_max=10000):
    # Save input data
    np.save("input/A.npy", A)
    np.save("input/B.npy", B)

    # Run the kernel with correct arguments
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    cmd = [
        f"./{kernel}",
        "--input",
        "input",
        "--output",
        "output",
        "--trial-max",
        str(trials_max),
        "--time-max",
        str(max_time),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {kernel}:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Kernel {kernel} execution failed")

    C = np.load("output/C.npy")
    with open("output/measurements.json", "r") as f:
        measurements = dict(json.load(f))
    measurements["C"] = C
    return measurements


def benchmark_by_size(max_speed_gflops, naive_kernel_name, kernel_list, num_threads=12):
    test_sizes = [
        31,
        32,
        96,
        97,
        127,
        128,
        129,
        191,
        192,
        229,
        255,
        256,
        257,
        319,
        320,
        321,
        417,
        479,
        480,
        511,
        512,
        639,
        640,
        767,
        768,
        769,
    ]
    
    all_kernel_gflops = {}
    
    for n in test_sizes:
        # Double precision float
        # Ref: https://numpy.org/doc/stable/user/basics.types.html#relationship-between-numpy-data-types-and-c-data-types
        input_data_A = np.random.rand(n, n).astype(np.float64)
        input_data_B = np.random.rand(n, n).astype(np.float64)

        baseline_result = run_dgemm(
            naive_kernel_name, input_data_A, input_data_B, num_threads=1
        )

        baseline_gflops = 2.0e-9 * n * n * n / (baseline_result["time"] * 1.0e-9)
        all_kernel_gflops.setdefault(naive_kernel_name, []).append(baseline_gflops)

        for kernel_name in kernel_list:
            optimized_result = run_dgemm(
                kernel_name, input_data_A, input_data_B, num_threads=num_threads
            )
            check_correctness(input_data_A, input_data_B, optimized_result["C"])

            optimized_gflops = 2.0e-9 * n * n * n / (optimized_result["time"] * 1.0e-9)
            optimized_peak_perc = (optimized_gflops / max_speed_gflops) * 100

            print(
                f"{kernel_name:18s}: Size: {n:4d} GFLOPS: {optimized_gflops:5.2f} %peak: {optimized_peak_perc:4.2f}% speedup: {baseline_result['time'] / optimized_result['time']:5.2f}x"
            )
            
            all_kernel_gflops.setdefault(kernel_name, []).append(optimized_gflops)

    draw_and_save_plot(test_sizes, all_kernel_gflops, "Matrix Size", "GFLOP/s", "GFLOP/s for MatMul on matrices of varying sizes", "benchmark_comparison.png")

def benchmark_strong_scaling(kernel_list, matrix_size, max_num_threads):
    thread_counts = [i for i in range(1,max_num_threads + 1)]
    
    input_data_A = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    input_data_B = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    
    all_kernel_speedups = {}
    
    for kernel_name in kernel_list:
        print(f"\n=== Strong Scaling for kernel: {kernel_name} ===")
        speedup = []

        # Single thread performance
        single_thread_result = run_dgemm(
            kernel_name, input_data_A, input_data_B, num_threads=1
        )
        
        for thread_count in thread_counts:
            result = run_dgemm(
                kernel_name, input_data_A, input_data_B, num_threads=thread_count
            )
            check_correctness(input_data_A, input_data_B, result["C"])
            speedup.append(single_thread_result["time"]/result["time"])
            print(
                f"{kernel_name:18s}: Threads: {thread_count:4d} Size: {matrix_size:4d} speedup: {single_thread_result['time'] / result['time']:5.2f}x"
            )

        all_kernel_speedups[kernel_name] = speedup

    draw_and_save_plot(thread_counts, all_kernel_speedups, "Number of threads", "Speedup over single thread", f"Strong Scaling Plot for {matrix_size}x{matrix_size} MatMul", "strong_scaling_comparison.png")


def benchmark_weak_scaling(kernel_list, first_matrix_size, max_num_threads):
    all_kernel_speedups = {}
    
    for kernel_name in kernel_list:
        print(f"\n=== Weak Scaling for kernel: {kernel_name} ===")
        speedup = []
        first_time = None

        for thread_count in range(1, max_num_threads + 1):
            test_size = np.ceil(first_matrix_size * np.sqrt(thread_count)).astype(int)
            input_data_A = np.random.rand(test_size, test_size).astype(np.float64)
            input_data_B = np.random.rand(test_size, test_size).astype(np.float64)

            result = run_dgemm(
                kernel_name, input_data_A, input_data_B, num_threads=thread_count
            )
            check_correctness(input_data_A, input_data_B, result["C"])

            if first_time is None:
                first_time = result["time"]

            print(
                f"{kernel_name:18s}: Threads: {thread_count:4d} Size: {test_size:4d} speedup: {first_time / result['time']:5.2f}x"
            )
                
            speedup.append(first_time/result["time"])

        all_kernel_speedups[kernel_name] = speedup

    draw_and_save_plot(
        [i for i in range(1, max_num_threads + 1)],
        all_kernel_speedups, 
        "Number of threads", 
        "Speedup over single thread", 
        f"Weak Scaling Plot for MatMul matrices {first_matrix_size}-{np.ceil(first_matrix_size * np.sqrt(max_num_threads)).astype(int)}", 
        "weak_scaling_comparison.png"
    )



if __name__ == "__main__":
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("plot", exist_ok=True)

    # The reference machine is Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz (turbo boost is disabled)
    # Manual: https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-70-ghz/specifications.html
    # VPUs: https://cvw.cac.cornell.edu/vector/hardware/vector-processing-unit#:~:text=Vector%20processing%20units%20(VPUs)%20perform%20the%20actual,are%20equipped%20with%20two%20VPUs%20per%20core.
    # 8 (vectorization width) x 2 (vector processing units) x 2 (FMA units)
    max_speed_gflops = 2.7 * 8 * 2 * 2

    naive_kernel_name = "dgemm-naive"
    optimized_kernel_name = "dgemm-optimized"

    max_num_threads = 12
    parser = argparse.ArgumentParser(description="Benchmark matrix multiplication kernels")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark over varying matrix sizes")
    parser.add_argument("--strong-scaling", action="store_true", help="Run strong scaling benchmark on a fixed matrix size")
    parser.add_argument("--weak-scaling", action="store_true", help="Run weak scaling benchmark starting from a small matrix size")
    parser.add_argument("--kernels", nargs='*', type=str, default=[optimized_kernel_name], help="versions to benchmark")
    
    args = parser.parse_args()
    
    if not any([args.benchmark, args.strong_scaling, args.weak_scaling]):
        args.benchmark = True
    
    if args.benchmark:
        benchmark_by_size(max_speed_gflops, naive_kernel_name, args.kernels)
    
    if args.strong_scaling:
        matrix_size = 768
        benchmark_strong_scaling(args.kernels, matrix_size, max_num_threads)

    if args.weak_scaling:
        first_matrix_size = 222
        benchmark_weak_scaling(args.kernels, first_matrix_size, max_num_threads)