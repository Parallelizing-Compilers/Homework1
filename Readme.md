# Homework 1: Optimizing Matrix Multiplication

**Due Date:** Tuesday, February 6th at 11:59 PM PST

(Content adapted with permission from [CS267 Spring 2025: Optimizing Matrix Multiplication](https://sites.google.com/lbl.gov/cs267-spr2025/hw-1?authuser=0#h.p_wUx8PSDWAdpd))

## Table of Contents
* [Problem statement](#problem-statement)
* [Instructions](#instructions)
    * [Teams](#teams)
    * [Getting Started with PACE](#getting-started-with-pace)
    * [Getting Set Up](#getting-set-up)
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

Your task in this assignment is to write an optimized [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) function for ICE's supercomputer. We will give you a generic matrix multiplication code (also called matmul or dgemm), and it will be your job to tune our code to run efficiently on ICE's processors. We are asking you to write an optimized single-threaded matrix multiply kernel. This will run on only one core.

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

The starter code is available on GitHub at [https://github.com/Parallelizing-Compilers/Homework1](https://github.com/Parallelizing-Compilers/Homework1) and should work out of the box. To get started, we recommend you log in to ICE and download the assignment. This will look something like the following:

## Getting Started with PACE

Please read through the PACE tutorial, available here: [https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102)


```
student@local:~> ssh <gatech_username>@login-ice.pace.gatech.edu
student@login-ice-1:~> git clone [https://github.com/Parallelizing-Compilers/Homework1.git](https://github.com/Parallelizing-Compilers/Homework1.git)
student@login-ice-1:~> salloc -N 1 -n 12 -t 04:00:00 -C gold6226
student@login-ice-1:~> cd Homework1
student@login-ice-1:~> ls
benchmark.cpp  dgemm-naive.c      json.hpp  npy.hpp         Readme.md
benchmark.py   dgemm-optimized.c  Makefile  pyproject.toml  requirements.txt
```


* **Makefile**


The build script that manages compiling your code.
* **Readme.md**


This Readme
* **benchmark.py**

A driver program that runs your code.
* **benchmark.cpp**, **npy.hpp**, **json.hpp**

Helpers for benchmark.py

* **dgemm-optimized.c**  - - -  **You may only modify this file.** 

A simple blocked implementation of matrix multiply. It is your job to optimize the `square_dgemm()` function in this file.
* **dgemm-naive.c** 

For illustrative purposes, a naive implementation of matrix multiply using three nested loops.


> Please **do not** modify any of the files besides _dgemm-optimized.c_


## Installation
- Use pip to install the dependencies, ensure that you have Python 3.12 or greater.
    ```bash
    module load python/3.12.5
    python -m venv hw1
    source hw1/bin/activate
    pip install -r requirements.txt
    ```

- Ensure you have a C compiler (e.g., `gcc`) installed to compile the benchmarking harness.

- Build the C benchmarking harness by running `make` in the terminal.

## Interactive Session

You may find it useful to launch an [interactive session](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096#interactive-jobs) when developing your code. This lets you compile and run code interactively on a compute node that you've reserved. In addition, running interactively lets you use the special interactive queue, which means you'll receive your allocation quicker. For example: 

```
student@local:~> ssh <gatech_username>@login-ice.pace.gatech.edu
student@login-ice-1:~> salloc -N 1 -n 12 -t 04:00:00 -C gold6226
student@login-ice-1:~> cd Homework1
student@login-ice-1:~> module load python/3.12.5
student@login-ice-1:~> python -m venv hw1
student@login-ice-1:~> source hw1/bin/activate
student@login-ice-1:~/Homework1> python benchmark.py
```

## Editing the Code

One of the easiest ways to implement your homework is to directly change the code on the server. For this you need to use a command line editor like `nano` or `vim`.

For beginners we recommend taking your first steps with `nano`. You can use it on PACE like this:

```
student@login-ice-1:~/HW1> module load nano
student@login-ice-1:~/HW1> nano dgemm-optimized.c
```
Use `Ctrl+X` to exit.

For a more complete code editor try _vim_ which is loaded by default:
```
student@login-ice-1:~/HW1> vim dgemm-optimized.c
```
Use `Esc` and `:q` to exit. (`:q!` if you want to discard changes). Try out the [interactive vim tutorial](https://www.openvim.com/) to learn more.

Using hosted version control like GitHub makes uploading your changes much easier. If you're in a Windows environment, consider using the Windows Subsystem for Linux (WSL) for development.

## Standard Processor On PACE

On Pace ICE, we use any Dual Xeon Gold 6226 processor (see all the resources here): [https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095).


The reference machine is Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz (turbo boost is disabled). The manual is here:https://www.intel.com/content/www/us/en/products/sku/193957/intel-xeon-gold-6226-processor-19-25m-cache-2-70-ghz/specifications.html
there are two [VPUs](https://cvw.cac.cornell.edu/vector/hardware/vector-processing-unit#:~:text=Vector%20processing%20units%20(VPUs)%20perform%20the%20actual,are%20equipped%20with%20two%20VPUs%20per%20core) per core, each with 512-bit vector width, so 8 double-precision (64-bit) elements can be processed in parallel. Each VPU has FMA units.

You can learn more about the cpu by running the following command on PACE:
```
student@login-ice-1:~> lscpu
```

## Our Harness

The `benchmark.py` file generates matrices of a number of different sizes and benchmarks the performance. It outputs the performance in [FLOPS](https://en.wikipedia.org/wiki/FLOPS) and in a percentage of theoretical peak attained. Your job is to get your matrix-multiply's performance as close to the theoretical peak as possible.

### Theoretical Peak

Our benchmark harness reports numbers as a percentage of theoretical peak. Here, we show you how we calculate the theoretical peak. If you'd like to run the assignment on your own processor, you should follow this process to arrive at the theoretical peak of your own machine, and then replace the **max_speed** constant in `benchmark.py` with the theoretical peak of your machine. Be sure to change it back if you run your code on PACE again.

### One Core

One core has a normal clock rate of 2.7 GHz, so it can issue 2.7 billion instructions per second at maximum. Our processors also have a 512-bit _vector width_, meaning each instruction can operate on 8 64-bit data elements at a time. Furthermore, the processor includes a _fused multiply-add_ (FMA) instruction, which means 2 floating point operations can be performed in a single instruction.

So, the theoretical peak is:
- 2.7 GHz * 8-element (512-bit) vector * 2 vector pipelines * 2 ops in an FMA = 86.4 GFlops/s

### Optimizing

Now, it's time to optimize!  A few optimizations you might consider adding:
1. Perform blocking. The dgemm-optimized.c already gets you started with this, although you'll need to tune block sizes.
2. Write a register-blocked kernel, either by writing an inner-level fixed-size matrix multiply and hoping (and [maybe checking](https://godbolt.org/)) that the compiler inlines it, writing [AVX intrinsics](https://software.intel.com/sites/landingpage/IntrinsicsGuide/), or even writing inline assembly instructions.
3. Add manual prefetching.

You may, of course, proceed however you wish.  We recommend you look through the lecture notes as reference material to guide your optimization process, as well as the references at the bottom of this write-up.

# Notes

* **Your grade will mostly depend on three factors:**
    * whether or not it is correct (ie. finishes running without exiting early)
    * performance sustained by your codes on the ICE supercomputer,
    * explanations of the performance features you observed (including what didn't work)
* There are other formulations of matmul (e.g., [Strassen](http://en.wikipedia.org/wiki/Strassen_algorithm)) that are mathematically equivalent, but perform asymptotically fewer computations - we will not grade submissions that do fewer computations than the 2n^3 algorithm. This is actually an optional part of HW1.
* You must use the GNU C Compiler 12.3 for this assignment. If your code does not compile and run with GCC 12.3, it will not be graded.
* Besides compiler intrinsic functions and built-ins, your code (`dgemm-optimized.c`) must only call into the C standard library.
* GNU C provides [many](http://gcc.gnu.org/onlinedocs/gcc/C-Extensions.html) extensions, which include intrinsics for vector (SIMD) instructions and data alignment. (Other compilers may have different interfaces.)
    * To manually vectorize, you should prefer to add compiler intrinsics to your code; avoid using inline assembly, at least at first.
    * The [Compiler Explorer](https://gcc.godbolt.org/z/v2fTDJ) project will be useful for exploring the relationship between your C code and its corresponding assembly. Release mode builds compile with `-O3`.
* You may assume that A and B do not alias C; however, A and B may alias each other. It is semantically correct to qualify C (the last argument to square_dgemm) with the C99 `restrict` keyword. There is a lot online about restrict and pointer-aliasing - [this](http://cellperformance.beyond3d.com/articles/2006/05/demystifying-the-restrict-keyword.html) is a good article to start with, along with the [Wikipedia article](https://en.wikipedia.org/wiki/Restrict) on the restrict keyword.
* The matrices are all stored in [column-major order](http://en.wikipedia.org/wiki/Row-major_order), i.e. `C[i,j] == C(i,j) == C[(i-1)+(j-1)*n]`, for `i=1:n`, where `n` is the number of rows in C. Note that we use 1-based indexing when using mathematical symbols and MATLAB index notation (parentheses), and 0-based indexing when using C index notation (square brackets).
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
* [GCC](https://gcc.gnu.org/onlinedocs/gcc-11.2.0/gcc/) documentation - Perlmutter's default version currently is GCC 12.3.0.
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
    * (Note: PHiPAC is a code-generating autotuner for matmul that started as a submission for this HW in a previous semester of CS267. Also see [ATLAS](http://math-atlas.sourceforge.net/); both are good examples if you are considering code generation strategies.)
* Lam, M. S., Rothberg, E. E, and Wolf, M. E. 1991. The Cache Performance and Optimization of Blocked Algorithms, ASPLOS'91, 63-74.
    * (Note: clearly explains cache blocking, supported by with performance models.)