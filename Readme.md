# Homework 1: Optimizing Matrix Multiplication

**Due Date:** Tuesday, February 6th at 11:59 PM PST

(Content adapted with permission from [CS267 Spring 2025: Optimizing Matrix Multiplication](https://sites.google.com/lbl.gov/cs267-spr2025/hw-1?authuser=0#h.p_wUx8PSDWAdpd))

## Table of Contents
* [Problem statement](#problem-statement)
* [Instructions](#instructions)
    * [Teams](#teams)
    * [Getting Started with PACE](#getting-started-with-pace)
    * [Building our Code](#building-our-code)
    * [Running our Code](#running-our-code)
    * [Interactive Session](#interactive-session)
    * [Editing the Code](#editing-the-code)
    * [Our Harness](#our-harness)
    * [Standard Processor On PACE](#Standard-Processor-On-PACE)
* [Grading](#grading)
    * [Submission Details](#submission-details)
    * [Write-up Details](#write-up-details)
* [Notes](#notes)
* [Optional Parts](#optional-parts)
* [Documentation](#documentation)
* [References](#references)

---

## Problem statement

Your task in this assignment is to write an optimized [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) function for the PACE ICE Cluster. We will give you a generic matrix multiplication code (also called matmul or dgemm), and it will be your job to tune our code to run efficiently on the Intel Xeon Gold 6226 processors. We are asking you to write an optimized matrix multiply kernel which can be multi-threaded and run on multiple cores.

We consider a special case of matmul:

_C := C + A*B_

where A, B, and C are n x n matrices. This can be performed using 2n^3 floating point operations (n^3 adds, n^3 multiplies), as in the following pseudocode:

```
for i = 1 to n
  for j = 1 to n
    for k = 1 to n
      C(i,j) = C(i,j) + A(i,k) * B(k,j)
    end
  end
end
```

---

# Instructions

## Getting Set Up

The starter code is available on GitHub at [https://github.com/Parallelizing-Compilers/Homework1](https://github.com/Parallelizing-Compilers/Homework1) and should work out of the box. To get started, we recommend you log in to PACE-ICE and download the assignment.

## Important Note :
Tasks requiring heavy compute on the login node will be automatically killed and may result in account suspension. You must use salloc to allocate a compute node before running this assignment.

## Getting Started with PACE

If you are new to the PACE cluster, please ensure you are connected to the GT VPN. You will be logging into the ICE cluster environment.

Please read through the PACE tutorial, available here: [https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102)

To get started, log in to the PACE login node and clone the assignment.
```
student@local:~> ssh <gatech_username>@login-ice.pace.gatech.edu
student@login-ice-1:~> cd scratch
student@login-ice-1:~> git clone [https://github.com/Parallelizing-Compilers/Homework1.git](https://github.com/Parallelizing-Compilers/Homework1.git)
student@login-ice-1:~> cd Homework1
student@login-ice-1:~> ls
benchmark.cpp  dgemm-naive.c      json.hpp  npy.hpp         Readme.md
benchmark.py   dgemm-optimized.c  Makefile  pyproject.toml  requirements.txt
```

There are ten files in the base repository. Their purposes are as follows:

* **Makefile**

The build script that manages compiling your code.

* **benchmark.cpp**

This file benchmarks the matrix multiplication and saves the result and timings.

* **Readme.md**

This Readme

* **npy.hpp**, **json.hpp**

Helpers

* **benchmark.py**

This executes your pre-compiled binaries to measure performance, verify correctness, and generate scaling plots.

* **dgemm-optimized.c**  - - -  **Only this file and your report pdf will be graded.** 

A simple blocked implementation of matrix multiply. It is your job to optimize the `square_dgemm()` function in this file.

* **dgemm-naive.c** 

For illustrative purposes, a naive implementation of matrix multiply using three nested loops.


## Moving to a Compute Node

Command to request an [interactive session](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096#interactive-jobs): [We will be making use of Intel Xeon Gold 6226 processor]
```
salloc -N 1 -n 1 -c <no. of cores> -t <session-time> -C gold6226
```

Once the command is granted, your terminal prompt will change (e.g., to [student@atl1-1-02-003-19-2]$). You are now on a compute node.

## Building the Code

Once you are on a compute node (inside salloc):

We use a Makefile to simplify compilation.

1. Load modules
```
student@atl1-1-02-003-19-2:~> module load gcc
student@atl1-1-02-003-19-2:~> module load anaconda3
```

2. Clean previous builds if any and then compile
```
student@atl1-1-02-003-19-2:~> make clean
student@atl1-1-02-003-19-2:~> make
```
This will generate two executables: dgemm-naive and dgemm-optimized.

3. Load Python Environment
```
student@atl1-1-02-003-19-2:~> python -m venv hw1
student@atl1-1-02-003-19-2:~> source hw1/bin/activate
```

## Running the Code
Once you are on a compute node (inside salloc):

2. Run the Python Driver
```
student@atl1-1-02-003-19-2:~> python3 benchmark.py
```
Running this script will verify your implementation's correctness, reporting per-size performance metrics (GFLOPS & Speedup) to the terminal, and saves the final scaling graph to the plot/ directory.

Usage:
```
python3 benchmark.py [--help] [--benchmark] [--strong-scaling] [--weak-scaling]
    --help            : Show this help message and exit
    --benchmark       : Run benchmark over varying matrix sizes
    --strong-scaling  : Run strong scaling benchmark on a fixed matrix size
    --weak-scaling    : Run weak scaling benchmark starting from a small matrix size
```


## Editing the Code

One of the easiest ways to implement your homework is to directly change the code on the server. For this you need to use a command line editor like `nano` or `vim`.

For beginners we recommend taking your first steps with `nano`. You can use it on PACE like this:

```
student@atl1-1-02-003-19-2:~> module load nano
student@atl1-1-02-003-19-2:~> nano dgemm-optimized.c
```
Use `Ctrl+X` to exit.

For a more complete code editor try _vim_ which is loaded by default:
```
student@login-ice-1:~/HW1> vim dgemm-optimized.c
```
Use `Esc` and `:q` to exit. (`:q!` if you want to discard changes). Try out the [interactive vim tutorial](https://www.openvim.com/) to learn more.

Using hosted version control like GitHub makes uploading your changes much easier. If you're in a Windows environment, consider using the Windows Subsystem for Linux (WSL) for development.

## Our Harness

The `benchmark.py` file generates matrices of a number of different sizes and benchmarks the performance. It outputs the performance in [FLOPS](https://en.wikipedia.org/wiki/FLOPS) and in a percentage of theoretical peak attained. Your job is to get your matrix-multiply's performance as close to the theoretical peak as possible.

## Standard Processor On PACE

On Pace ICE, we use any Dual Xeon Gold 6226 processor (see all the resources here): [https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095).


The reference machine is Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz (turbo boost is disabled). The manual is here:https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-70-ghz/specifications.html
there are two [VPUs](https://cvw.cac.cornell.edu/vector/hardware/vector-processing-unit#:~:text=Vector%20processing%20units%20(VPUs)%20perform%20the%20actual,are%20equipped%20with%20two%20VPUs%20per%20core) per core, each with 512-bit vector width, so 8 double-precision (64-bit) elements can be processed in parallel. Each VPU has FMA units.

You can learn more about the cpu by running the following command on PACE:
```
student@login-ice-1:~> lscpu
```


### Theoretical Peak

Our benchmark harness reports numbers as a percentage of theoretical peak. Here, we show you how we calculate the theoretical peak. If you'd like to run the assignment on your own processor, you should follow this process to arrive at the theoretical peak of your own machine, and then replace the **max_speed** constant in `benchmark.py` with the theoretical peak of your machine. Be sure to change it back if you run your code on PACE again.

### Single Core Peak

One core has a normal clock rate of 2.7 GHz, so it can issue 2.7 billion instructions per second at maximum. Our processors also have a 512-bit _vector width_, meaning each instruction can operate on 8 64-bit data elements at a time. Furthermore, the processor includes a _fused multiply-add_ (FMA) instruction, which means 2 floating point operations can be performed in a single instruction.

So, the theoretical peak is:
- 2.7 GHz * 8-element (512-bit) vector * 2 vector pipelines * 2 ops in an FMA = 86.4 GFlops/s

### Multi Core Peak 

You can use multiple cores in this assignment and for that the calculation will be as below :
- If you use 4 cores: $86.4 \times 4 = \mathbf{345.6 \text{ GFLOPS}}$

### Optimizing

Now, it's time to optimize!  A few optimizations you might consider adding:
1. Perform blocking. Break the matrix into smaller sub-matrices that fit into L1/L2 cache. The dgemm-optimized.c already gets you started with this, although you'll need to tune block sizes.
2. Write a register-blocked kernel, either by writing an inner-level fixed-size matrix multiply and hoping (and [maybe checking](https://godbolt.org/)) that the compiler inlines it, writing [AVX intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/), or even writing inline assembly instructions.
3. Multithreading: Use OpenMP to utilize all cores on the node.

You may, of course, proceed however you wish.  We recommend you look through the lecture notes as reference material to guide your optimization process, as well as the references at the bottom of this write-up.

# Grading

We will grade your assignment by reviewing your write-up, analyzing the optimizations you attempted in _dgemm-optimized.c_, and benchmarking your code's performance on the PACE cluster. Note that code that returns incorrect results will receive significant penalties.

## Submission Details

1.  Ensure that your write-up is located in your source directory, next to **dgemm-optimized.c**. It should be named **cs6245_<gt_username>_hw1.pdf**.
2.  Clean your build directory 
    ```
    make clean
    ```
    This second command will fail if the PDF is not present.
3.  Create a compressed archive of your work:
    ```
    tar -czvf cs6245_<gt_username>_hw1_submission.tar.gz dgemm-optimized.c cs6245_<gt_username>_hw1.pdf Makefile
    ```
4.  Submit the .tar.gz file through Canvas

## Write-up Details

* Your write-up should contain:
    * The optimizations used or attempted,
    * the results of those optimizations (a plot with different lines for different versions),
    * the reason for any odd behavior (e.g., dips) in performance

* Your write-up should be a maximum of 3 pages in length, including all text, figures, tables, and references.

# Notes

* **Your grade will mostly depend on three factors:**
    * Whether or not it is correct (ie. finishes running without exiting early)
    * Performance sustained on the Intel Xeon Gold 6226.
    * Explanations of the performance features you observed (including what didn't work)
    
* There are other formulations of matmul (e.g., [Strassen](http://en.wikipedia.org/wiki/Strassen_algorithm)) that are mathematically equivalent, but perform asymptotically fewer computations - we will not grade submissions that do fewer computations than the 2n^3 algorithm. This is actually an optional part of HW1.
* You must use the GNU C Compiler for this assignment. If your code does not compile and run with GCC, it will not be graded.
* Besides compiler intrinsic functions and built-ins, your code (`dgemm-optimized.c`) must only call into the C standard library.
* GNU C provides [many](http://gcc.gnu.org/onlinedocs/gcc/C-Extensions.html) extensions, which include intrinsics for vector (SIMD) instructions and data alignment. (Other compilers may have different interfaces.)
    * To manually vectorize, you should prefer to add compiler intrinsics to your code; avoid using inline assembly, at least at first.
    * The [Compiler Explorer](https://gcc.godbolt.org/z/v2fTDJ) project will be useful for exploring the relationship between your C code and its corresponding assembly. Release mode builds compile with `-O3`.
* You may assume that A and B do not alias C; however, A and B may alias each other. It is semantically correct to qualify C (the last argument to square_dgemm) with the C99 `restrict` keyword. There is a lot online about restrict and pointer-aliasing - [this](http://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html) is a good article to start with, along with the [Wikipedia article](https://en.wikipedia.org/wiki/Restrict) on the restrict keyword.
* The matrices are all stored in [row-major order](http://en.wikipedia.org/wiki/Row-major_order), i.e. `C[i,j] == C(i,j) == C[(i-1)*n + (j-1)]`, for `i=1:n`, where `n` is the number of rows in C. Note that we use 1-based indexing when using mathematical symbols and MATLAB index notation (parentheses), and 0-based indexing when using C index notation (square brackets).
* We will check correctness by the following component-wise error bound: |square_dgemm(n,A,B,0) - A*B| < eps*n*|A|*|B|.
    * where eps := 2^-52 = 2.2 * 10^-16 is the [machine epsilon](http://en.wikipedia.org/wiki/Machine_epsilon).

---

# Optional Parts

These parts are not graded. You should be satisfied with your square_dgemm results and write-up before beginning an optional part.

* Implement Strassen matmul. Consider switching over to the three-nested-loops algorithm when the recursive subproblems are small enough.
* Support the dgemm interface (ie, rectangular matrices, transposing, scalar multiples).
* Try float (single-precision).
* Try complex numbers (single- and double-precision) - note that complex numbers are part of C99 and [supported in gcc](http://gcc.gnu.org/onlinedocs/gcc/Complex.html). [This forum thread](http://stackoverflow.com/questions/3211346/complex-mul-and-div-using-sse-instructions) gives advice on vectorizing complex multiplication with the conventional approach - but note that there are [other algorithms](http://en.wikipedia.org/wiki/Multiplication_algorithm#Gauss.27s_complex_multiplication_algorithm) for this operation.
* Optimize your matmul for the case when the inputs are symmetric. Consider [conventional](http://www.netlib.org/lapack/lug/node122.html) and [packed](http://www.netlib.org/lapack/lug/node123.html) symmetric storage.

---

# Documentation

* [ICE's programming environment](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102) documentation
* [GCC](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/) documentation
* [Intel's intrinsics guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX2,FMA) - a complete overview of all available vector intrinsics.
* [GCC's vector extensions](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/Vector-Extensions.html#Vector-Extensions) - special types that make programming with vectors easier
* [GCC's built-ins](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/Other-Builtins.html#Other-Builtins) - special commands that give optimization hints to the compiler. See assume_aligned, unreachable, and expect. [Some are specific to x86.](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/x86-Built-in-Functions.html#x86-Built-in-Functions)
* [GCC's variable attributes](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/Common-Variable-Attributes.html#Common-Variable-Attributes) - useful for optimizing the memory layout of your program
* [GCC's function attributes](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/Common-Function-Attributes.html#Common-Function-Attributes) - useful for controlling the optimization of particular functions. Some are [specific to x86.](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/x86-Function-Attributes.html#x86-Function-Attributes)

You are also welcome to learn from the source code of state-of-art BLAS implementations such as [ATLAS](http://math-atlas.sourceforge.net/). However, you should not reuse those codes in your submission.

Please only use one node at a time to conserve resources for other users.
---

# References

* Goto, K., and van de Geijn, R. A. 2008. Anatomy of High-Performance Matrix Multiplication, ACM Transactions on Mathematical Software 34, 3, Article 12.
    * (Note: explains the design decisions for the GotoBLAS dgemm implementation, which also apply to your code.)
* Chellappa, S., Franchetti, F., and Puschel, M. 2008. [How To Write Fast Numerical Code: A Small Introduction](https://users.ece.cmu.edu/~franzf/papers/gttse07.pdf), Lecture Notes in Computer Science 5235, 196-259.
    * (Note: how to write C code for modern compilers and memory hierarchies, so that it runs fast. Recommended reading, especially for newcomers to code optimization.)
* Bilmes, et al. [The PHiPAC (Portable High Performance ANSI C) Page for BLAS3 Compatible Fast Matrix Matrix Multiply](https://people.eecs.berkeley.edu/~krste/papers/phipac_ics97.pdf).
    * Also see [ATLAS](http://math-atlas.sourceforge.net/)
* Lam, M. S., Rothberg, E. E, and Wolf, M. E. 1991. The Cache Performance and Optimization of Blocked Algorithms, ASPLOS'91, 63-74.
    * (Note: clearly explains cache blocking, supported by with performance models.)