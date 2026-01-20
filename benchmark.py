import json
import os
import numpy as np
import scipy
import shutil
import subprocess
import matplotlib.pyplot as plt


def draw_and_save_plot(x, y, x_label, y_label, plot_title, file_name):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="s", color="green")

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)

    plt.savefig(f"plot/{file_name}", dpi=300, bbox_inches="tight")


def run_dgemm(kernel, A, B, n_trials=10, num_threads=1):
    # Save input data
    np.save("input/A.npy", A)
    np.save("input/B.npy", B)

    # Run the kernel with correct arguments
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    cmd = [
        f"./{kernel}",
        "--input",
        "input",
        "--output",
        "output",
        "--trial-max",
        str(n_trials),
        "--time-max",
        "inf",
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


def benchmark_single_thread(max_speed_gflops, naive_kernel_name, optimized_kernel_name):
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
    measured_kernel_gflops = []

    for n in test_sizes:
        # Double precision float
        # Ref: https://numpy.org/doc/stable/user/basics.types.html#relationship-between-numpy-data-types-and-c-data-types
        input_data_A = np.random.rand(n, n).astype(np.float64)
        input_data_B = np.random.rand(n, n).astype(np.float64)

        baseline_times = []
        optimized_times = []

        n_trials = 10
        for trial in range(n_trials):
            # Randomly decide order for this trial to avoid systemic bias
            order = np.random.choice(["baseline_first", "optimized_first"])

            if order == "baseline_first":
                baseline_result = run_dgemm(
                    naive_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=1,
                )
                baseline_times.append(baseline_result["time"])
                optimized_result = run_dgemm(
                    optimized_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=1,
                )
                optimized_times.append(optimized_result["time"])
                assert np.allclose(
                    baseline_result["C"], optimized_result["C"]
                ), "Results do not match!"
            else:
                optimized_result = run_dgemm(
                    optimized_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=1,
                )
                optimized_times.append(optimized_result["time"])
                baseline_result = run_dgemm(
                    naive_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=1,
                )
                baseline_times.append(baseline_result["time"])
                assert np.allclose(
                    baseline_result["C"], optimized_result["C"]
                ), "Results do not match!"

        baseline_times = np.array(baseline_times)
        optimized_times = np.array(optimized_times)

        wins = np.sum(optimized_times < baseline_times)
        win_rate = wins / n_trials
        p_value = scipy.stats.binomtest(wins, n_trials, 0.5).pvalue

        baseline_times_min = np.min(baseline_times)

        optimized_times_min = np.min(optimized_times)
        optimized_gflops = 2.0e-9 * n * n * n / (optimized_times_min * 1.0e-9)
        optimized_peak_perc = (optimized_gflops / max_speed_gflops) * 100

        print(
            f"Size: {n}    Gflops: {optimized_gflops:.2f}    peak perc: {optimized_peak_perc:.6f}%    speedup: {baseline_times_min / optimized_times_min:.2f}x"
        )

        measured_kernel_gflops.append(optimized_gflops)

    draw_and_save_plot(
        test_sizes,
        measured_kernel_gflops,
        "Matrix Size",
        "GFLOP/s",
        "GFLOP/s for MatMul on matrices of varying sizes",
        "single_thread.png",
    )


def benchmark_strong_scaling(optimized_kernel_name, max_num_threads):
    test_size = 512
    thread_counts = [i for i in range(1, max_num_threads)]
    speedup = []
    n_trials = 10

    input_data_A = np.random.rand(test_size, test_size).astype(np.float64)
    input_data_B = np.random.rand(test_size, test_size).astype(np.float64)

    # Single thread performance
    single_thread_result = run_dgemm(
        optimized_kernel_name,
        input_data_A,
        input_data_B,
        n_trials=n_trials,
        num_threads=1,
    )

    for thread_count in thread_counts:
        multithread_result = run_dgemm(
            optimized_kernel_name,
            input_data_A,
            input_data_B,
            n_trials=n_trials,
            num_threads=thread_count,
        )
        assert np.allclose(
            single_thread_result["C"], multithread_result["C"]
        ), "Results do not match!"
        speedup.append(single_thread_result["time"] / multithread_result["time"])

    draw_and_save_plot(
        thread_counts,
        speedup,
        "Number of threads",
        "Speedup over single thread",
        f"Strong Scaling Plot for {test_size}x{test_size} MatMul",
        "strong_scaling.png",
    )


def benchmark_weak_scaling(optimized_kernel_name, first_matrix_size, max_num_threads):
    thread_counts_test_size_tup = [
        (i, first_matrix_size * i) for i in range(1, max_num_threads)
    ]
    speedup = []
    n_trials = 10

    for thread_count, test_size in thread_counts_test_size_tup:
        input_data_A = np.random.rand(test_size, test_size).astype(np.float64)
        input_data_B = np.random.rand(test_size, test_size).astype(np.float64)

        single_thread_times = []
        multithread_times = []

        for _ in range(n_trials):
            # Randomly decide order for this trial to avoid systemic bias
            order = np.random.choice(["singlethread_first", "multithread_first"])

            if order == "singlethread_first":
                single_thread_result = run_dgemm(
                    optimized_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=1,
                )
                single_thread_times.append(single_thread_result["time"])
                multithread_result = run_dgemm(
                    optimized_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=thread_count,
                )
                multithread_times.append(multithread_result["time"])
                assert np.allclose(
                    single_thread_result["C"], multithread_result["C"]
                ), "Results do not match!"
            else:
                multithread_result = run_dgemm(
                    optimized_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=thread_count,
                )
                multithread_times.append(multithread_result["time"])
                single_thread_result = run_dgemm(
                    optimized_kernel_name,
                    input_data_A,
                    input_data_B,
                    n_trials=1,
                    num_threads=1,
                )
                single_thread_times.append(single_thread_result["time"])
                assert np.allclose(
                    single_thread_result["C"], multithread_result["C"]
                ), "Results do not match!"

        speedup.append(min(single_thread_times) / min(multithread_times))

    draw_and_save_plot(
        [tup[0] for tup in thread_counts_test_size_tup],
        speedup,
        "Number of threads",
        "Speedup over single thread",
        f"Weak Scaling Plot for MatMul matrices {first_matrix_size}-{first_matrix_size*max_num_threads}",
        "weak_scaling.png",
    )


if __name__ == "__main__":
    # Clean up and create directories
    if os.path.exists("input"):
        shutil.rmtree("input")
    if os.path.exists("output"):
        shutil.rmtree("output")
    if os.path.exists("plot"):
        shutil.rmtree("plot")

    os.makedirs("input")
    os.makedirs("output")
    os.makedirs("plot")

    # The reference machine is Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz (turbo boost is disabled)
    # Manual: https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-70-ghz/specifications.html
    # VPUs: https://cvw.cac.cornell.edu/vector/hardware/vector-processing-unit#:~:text=Vector%20processing%20units%20(VPUs)%20perform%20the%20actual,are%20equipped%20with%20two%20VPUs%20per%20core.
    # 8 (vectorization width) x 2 (vector processing units) x 2 (FMA units)
    max_speed_gflops = 2.7 * 8 * 2 * 2

    naive_kernel_name = "dgemm-naive"
    optimized_kernel_name = "dgemm-optimized"

    max_num_threads = 24

    # benchmark_single_thread(max_speed_gflops, naive_kernel_name, optimized_kernel_name)
    # benchmark_strong_scaling(optimized_kernel_name, max_num_threads)

    first_matrix_size = 32
    benchmark_weak_scaling(optimized_kernel_name, first_matrix_size, max_num_threads)
