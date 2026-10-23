---
id: unroll-your-loop
title: Unroll your loop | CUDA C++
date: Dec 10, 2025
snippet: Understanding `#pragma unroll` and how it helps performance.
tags: [cuda, gpu, performance]
---

> Understanding `#pragma unroll`

# What does unroll mean?

```cpp
// standard for loop
for(int i = 0; i<N; i++){
	A[i] += 0.1f;
}

// partially unrolled loop
for(int i = 0; i<N; i += 2){
	A[i] += 0.1f;
	A[i+1] = 0.1f;
}

//completely unrolled loop
A[0] += 0.1f;
A[1] += 0.1f;
.
.
.
A[N] += 0.1f;
```

In essence, you explicitly write down the loop operation, it may be in partial or full. Now to understand why this might be useful, we need to look into how loops are executed and what the compiler does at compile time.

![image.png](/blog/unroll-your-loop/image.png)

# Why unroll loops?

## Reduced Branch Overhead

Every loop iteration involves:

- Incrementing the loop counter
- Comparing the counter against the limit
- Conditional branching back to the loop start

These operations consume CPU/GPU cycles. By unrolling, you eliminate multiple branch instructions, reducing the overhead per operation.

## Better Instruction-Level Parallelism

Modern processors can execute multiple independent instructions simultaneously. When you unroll loops, you expose more independent operations that can be executed in parallel:

```cpp
// Original loop - operations are dependent
for(int i = 0; i < 4; i++){
    A[i] += B[i];
}

// Unrolled - all 4 operations are independent
A[0] += B[0];
A[1] += B[1];
A[2] += B[2];
A[3] += B[3];
```

The processor can potentially execute all four additions simultaneously if it has the resources.

## Improved Memory Access Patterns

Unrolling can help with:

- **Pre-fetching**: The processor can predict and load upcoming memory addresses
- **Cache utilization**: Sequential memory accesses are more cache-friendly
- **Register allocation**: The compiler can keep more values in fast registers

# What are Pragmas?

Pragma stands for "pragmatic information". It is a special instruction for the compiler, usually for optimization or platform-specific adjustments.

They follow the syntax:

```cpp
#pragma <instruction>
```

Pragmas are compiler-specific and are not part of the C++ standard. If a compiler does not recognize a particular `#pragma`, it will simply ignore it, and the code will still compile normally.

# The `#pragma unroll` Directive

In CUDA and modern C++ compilers, you don't need to manually unroll loops. The `#pragma unroll` directive tells the compiler to do it for you.

## Basic Usage

```cpp
// Fully unroll the loop
#pragma unroll
for(int i = 0; i < 4; i++){
    result += array[i];
}
```

## Partial Unrolling

```cpp
// Unroll by a factor of 4
#pragma unroll 4
for(int i = 0; i < N; i++){
    doSomething(i);
}
```

## CUDA Kernel Example

```cpp
__global__ void vectorAdd(float* A, float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 8
    for(int i = 0; i < 8; i++){
        int index = idx * 8 + i;
        if(index < N){
            C[index] = A[index] + B[index];
        }
    }
}
```

In this example, each thread processes 8 elements, and the loop is fully unrolled, generating 8 separate addition operations.

# Compiler Auto-Unrolling

Most modern compilers (GCC, Clang, MSVC, NVCC) automatically unroll loops when optimization is enabled (`-O2`, `-O3`). The `#pragma unroll` directive is mainly useful to:

- Force unrolling when the compiler is conservative
- Control the unroll factor explicitly
- Ensure consistent behavior across compilers
