---
id: the-cuda-worklog
title: The CUDA Worklog
date: Oct 23, 2025
snippet: Worklog as I learn CUDA and GPU Optimizations
tags: [cuda, gpu, performance]
---



# Day 1

## Vector Addition

“Straightforward”

You create an int variable which is basically the thread index, since we are launching enough threads, each thread does the computation for index, index + stride, index + 2*stride and so on.

The stride is essentially the number of total number of threads launched which can be calculated by blockDim.x * gridDim.x where blockDim.x tells you how many threads are in the block and gridDim.x tells you how many blocks are in the grid.

In a strict sense every kernel launch is essentially a grid created that handles computation. More on the architecture of GPU and how it relates to grid, blocks and threads.

```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //the parallelism is in the many threads; the loop is just a way to “tile” 
	    //the work across them.
    //the for loop is for a particular thread to handle computation
    //for example if you dont use the stride, and increment it simply by
	    // 1 then essentially
    //your entire computation is being overlapped by just each thread ie 
	    //every thread is doing the addition computation for all N elements
    for(int idx = i; idx<N; idx += stride){
        C[idx] = A[idx] + B[idx];
    }
}
```

## ReLU

“I didn’t want to do matmul next”

ReLU is essentially max(0, x), and given a 1D array of elements, all you have to do is write the same formula in the code. The rest of the kernel looks pretty much the same as the vector addition kernel. Since we launch enough threads -

thread 0 → 0, 0 + stride, 0 + 2*stride

thread 1 → 1, 1 + stride, 1 + 2*stride and so on

Similarly up until the nth thread. Like mentioned previously, the “for” loop in the kernel is to iterate through the elements that are computed per thread where the increment is equal to the stride. The parallelism is due to the fact that many threads are launched in parallel.

The initial kernel - 

```cpp
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int idx = i; idx < N; idx += stride){
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

// relu is simply max(0, x)
```

We use fmaxf for floating point values and fmax for double values. max is part of the cmath library in C++ and will not work inside a CUDA kernel. We may use fmax also for floating point values, I made that error initially and did not notice since the kernel passed the test results. Turns out that fmax will convert your float value to a double, perform the operation and then convert it back.

### surface level understanding of warp branching

Now the second question I had was, is this the fastest way? 

The first thing I checked was will a conditional statement like - convert to 0 if negative else leave as it is be faster. The short answer is no since it introduces branching (more on that later), and fmaxf is going to be faster because it is hardware intrinsic ie the operation is directly mapped to the hardware.

The long answer is it depends how you’ve structured your kernel code. If you’ve written something like a usual conditional statement - 

```cpp
for(int idx = i; idx<N; idx += stride){
	if(input[idx] < 0){
		output[idx] = 0;
	}
	else{
		output[idx] = input[idx];
	}
}
```

Then this causes something called warp branching. More on warps later, but the branching part is easy to understand. The conditional above, by nature has to be computed in a sequential manner. In CUDA, the instructions are executed in warp cycles. 32 threads are bundled under 1 warp and are supposed to execute the same instruction. Suppose when the kernel is launched and some threads take up the “if” branch and some threads take up the “else” branch, then branching occurs and one branch has to be temporarily disabled and can only continue once the “if” has been completed.

The alternate to this is to use “ternary operations” ie the symbols “?” and “:” to write a predicated statement that returns a value. This kernel should be equivalent to the “fmaxf” version.

```cpp
for(int idx = i; idx<N; idx += stride){
		output[idx] = (input[idx] < 0.0f) ? 0.0f : input[idx]; 
}
```

### speeding it up

→ float4 vectorization

## Color Inversion

→ see what is uchar4 for rgba

```cpp
__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int size = width * height * 4;
    for(int idx = i; idx < size; idx += stride ){
        if(idx % 4 != 3){
            image[idx] = 255 - image[idx];
        }
    }
}
```

# Day 2

## Matrix Transpose

Basic initial kernel - global memory usage with stride

```cpp
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; //column
    int j = blockDim.y * blockIdx.y + threadIdx.y; //row
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    
    for(int idx = j; idx < rows; idx += stride_y){
        for(int k = i; k < cols; k += stride_x){
            output[k * rows + idx] = input[idx * cols + k]; // row_num * width + col_num = i, j
        }
    }
}
//     while (i < cols && j < rows){
//         output[i * rows + j] = input[j * cols + i]
//     }
// }
}
```

### speeding it up

→ memory coalescing and tiling, shared memory access

# Day 3

### C/ C++ refresher

→ What is a C struct

→ Pointers to arrays

### Thread Organization

revisit threadIdx, blockIdx, blockDim

new understanding → a block can have 512 threads, now it is upto you how you arrange and access them, you may do (512, 1, 1), or (8,8,8) or  (16,16,2). So essentially, when you assign threadIdx.x or y or z to some int variable, you are setting how the threads in a block must have been arranged. This is simply for easy of use, the same kernel written with only threadIdx.x can be written in 2D access using .x and .y as well as 3D using .x .y .z 

![image.png](blog/cuda/CUDA/image.png)

__syncthreads()

![image.png](blog/cuda/CUDA/image%201.png)

transparent scalability in cuda makes the code not hardware bound, the grid is hypothetical and resource allocation adjusts according to the hardware underneath and not the code.

## Count Element

```cpp
__global__ void count_equal_kernel(const int* input, int* output, int N, int K) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int temp = 0;
    for(int idx = threadID; idx<N; idx += stride){
        temp += (input[idx] == K) ? 1 : 0;
    }
    atomicAdd(output, temp);
}
```

→ each thread checks the element corresponding to the arithmetic progression of indices incremented by stride (eg: 0, 0 + stride, 0 + stride*2) and increments the thread specific local variable temp. We then use atomicAdd to accumulate results from all local temp variables.

# Day 4

### Matrix Tranpose

```cpp
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; //column
    int j = blockDim.y * blockIdx.y + threadIdx.y; //row
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;
    
    for(int idx = j; idx < rows; idx += stride_y){
        for(int k = i; k < cols; k += stride_x){
            output[k * rows + idx] = input[idx * cols + k]; // row_num * width + col_num = i, j
        }
    }
}
```

### Reverse Array

```cpp
__global__ void reverse_array(float* input, int N) {
int threadId = blockDim.x * blockIdx.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;
for(int i = threadId; i < N/2; i+=stride){
    float temp = input[i];
    input[i] = input[N-i-1];
    input[N-i-1] = temp;
    }
}

```

# Day 5

### SiLU

```cpp
__global__ void silu_kernel(const float* input, float* output, int N) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = threadID; i<N; i+= stride){
        float x = __ldg(&input[i]);
        float neg_exp_x = __expf(-x);
        output[i] = x / (1 + neg_exp_x);
    }
}
```

→ __ldg

→ __expf

Initially I wrote code like how I would do math on paper, so I simplified the silu expression and took the exponent of negative power to a positive power in both numerator and denominator.

This caused an issue. For large positive values of x, expf(x) becomes quite large. On further multiplication with x in the numerator it can also lead to an overflow. expf(-x) in the denominator on other hand would tend to zero for a very large positive value of x. So what I learnt was that you’d rather let the formulation be as it is than to open it up.

Now, the difference between expf() and __expf(). The later is CUDA specific, hardware intrinsic, low- precision and faster. The former gives a full precision result.

### Count 2D

```cpp
__global__ void count_2d_equal_kernel(const int* input, int* output, int N, int M, int K) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; //columns
    int y = blockDim.y * blockIdx.y + threadIdx.y; //rows
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    int temp = 0;
    for(int i = x; i < M; i += stride_x){
        for(int j = y; j < N; j += stride_y){
            temp += (input[j * M + i] == K) ? 1 : 0;
        }
    }
    atomicAdd(output, temp);
}

```

### SwiGLU

```cpp
__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = threadId; i < halfN; i += stride){
        float x1 = __ldg(&input[i]);
        float x2 = __ldg(&input[i + halfN]);
        float neg_exp_x = __expf(-x1);
        float silu = x1 / (1.0f + neg_exp_x);
        output[i] = x2 * silu;
    }
}
```

### Copy Matrix

```cpp
__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int x = blockDim.x * blockIdx.x + threadIdx.x; //columns
    int y = blockDim.y * blockIdx.y + threadIdx.y; //rows
    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y; // it is an NxN matrix so equivalent strides in both dimensions I guess
    

    for(int j = y; j<N; j += stride_y){
        int row_offset = j * N; // Row offset saves multiplications in inner loop
        for(int i = x; i<N; i += stride_x){
            B[row_offset + i] = __ldg(&A[row_offset + i]);
        }
    }
}
```

Quite similar to count 1D and 2D or even matmul in the way we access elements of a matrix using 2 strided loops and calculating the index of the i,j element  because 2D arrays are flattened in memory. However, since we are copying the matrix here, instead of calculating the position of i,j using the formula j * num_cols + i (twice), we replace the term j * num_cols with a variable offset, and reuse this variable. As a result we have reduced one multiplication of recalculating the offset for both A and B.

### Rainbow Table

```cpp
#include <cuda_runtime.h>

__device__ unsigned int fnv1a_hash(int input) {
    const unsigned int FNV_PRIME = 16777619;
    const unsigned int OFFSET_BASIS = 2166136261;
    
    unsigned int hash = OFFSET_BASIS;
    
    for (int byte_pos = 0; byte_pos < 4; byte_pos++) {
        unsigned char byte = (input >> (byte_pos * 8)) & 0xFF;
        hash = (hash ^ byte) * FNV_PRIME;
    }
    
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = x; i<N; i += stride){
        int val = __ldg(&input[i]);
        for(int r = 0; r < R; r++){
            val = fnv1a_hash(val);
        }
        output[i] = val;
    }
}

```

→ gpt suggests to use #pragma unroll 8 and unsigned int (will check these out later)

# Day 6

## General Stuff

We now know what is float4 vectorization and also use it quite often.

Read half of Siboehm’s blog and we now understand that there are three key parts (my condensation of things I read) of performance optimization for a kernel

- the inherent parallelism (the usage of threads, the choice of using 1D or 2D thread indices with a 1D, 2D, or even a 3D kernel is entirely a design choice and you choose what you are comfortable with and what is the requirement of the given kernel
- memory access optimization, the way we access memory also affects the speed of the kernel, here comes something called coalescing, the basic idea behind this is to ensure that threads with sequential thread Ids access sequential memory on the gpu both during reading and writing, below is the visualization for a matmul kernel, one that I now use as a mental model for coalescing
- gpu occupancy - this involves some math to figure out what is the bottleneck to the kernel performance, whether it is the memory transfers or is it simply that we our not using our gpu to the fullest. we need to understand the warp grouping, the number of SMs, registers, max threads per block among other details to calculate the gpu occupancy. Siboehm’s blog is again a good read here
- when accessing threads via 3D indexing, the x dim is incremented first, then the y dim, then the z dim

# Day 7

## Tensara Exists

We find out that there exists a platform called tensara that gives a comprehensive benchmark for your kernels, allows you to run on B200 and has a public leaderboard, so we switch from leetgpu to tensara for a while.

The key while writing kernels here is that you also decide how you want to launch the kernel grid hence giving you more control.

The speed is measure in GFLOPS which stands for giga-floating point operations per second. The higher the better

### Vector Addition

427.19 GFLOPS on B200

```cpp
#include <cuda_runtime.h>

__global__ void vector_addition(const float4* __restrict__ d_input1,
                                const float4* __restrict__ d_input2,
                                float4* __restrict__ d_output,
                                size_t n4)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = x; i < n4; i += stride) {
        float4 a = d_input1[i];
        float4 b = d_input2[i];
        d_output[i] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
}

extern "C" void solution(const float* d_input1,
                         const float* d_input2,
                         float* d_output,
                         size_t n)
{
    size_t n4 = n / 4;  // number of float4-sized chunks

    // reinterpret float* as float4* (no copy)
    const float4* d_input1_4 = reinterpret_cast<const float4*>(d_input1);
    const float4* d_input2_4 = reinterpret_cast<const float4*>(d_input2);
    float4* d_output_4 = reinterpret_cast<float4*>(d_output);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (n4 + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = min(blocksPerGrid, 65535);

    vector_addition<<<blocksPerGrid, threadsPerBlock>>>(d_input1_4, d_input2_4, d_output_4, n4);

    // Handle remainder if n not multiple of 4
    size_t remainder = n4 * 4;
    if (n % 4 != 0) {
        // copy last few elements (n % 4 of them)
        // simple 1-thread kernel would be better for large data
        for (size_t i = remainder; i < n; ++i) {
            float a, b;
            cudaMemcpy(&a, d_input1 + i, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&b, d_input2 + i, sizeof(float), cudaMemcpyDeviceToHost);
            float sum = a + b;
            cudaMemcpy(d_output + i, &sum, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}
```

# Day 8

## Particle collision simulation with ML (in progress)

Decided to write a particle simulation with collision detection and use ML for the physics as an experiment instead.

This is still WIP. I have already trained an XGBoost, MLP in PyTorch and generated the data to train both of these using a pygame simulation. However my goal was to write this in cpp, load the models using something, and for MLP see if I can write a kernel to run it on GPU.

Additionally I also wanted to train a PINN for the same. Now in essence we would not be able to train a true PINN because the phenomenon of collision isnt continuous rather is an impulse and thus does not have a ordinary differential equation that can be incorporated in the loss function (which is the classical PINN method). However we can ground our model by using the laws of motions as regularizers in the loss function. Hence, we would use the facts that total momentum before and after collision must be conserved, this results in model predictions closer to true values.

# Day x

## Tensortonic

Somewhere in between Day8 and Day9 came up a site called TensorTonic, so we decieded to hop and try it out, it’s decent, got a bunch of questions, with the key being that it asks you to focus on vectorization.

We’re currently sitting at 12th on the leaderboard, with questions being added every few days, gotta take out time to do that.

Here’s a few select, mainly the ones that correspond to solid introductory concepts - 

[tensor_tonic/medium at main · grasgor/tensor_tonic](https://github.com/grasgor/tensor_tonic/tree/main/medium)

# Day 9

here’s a decent gap of probably 3-4 days because I was setting up my system, tinkering around with unnecessary customizations (also got to know that hyprland is close to non existent on ubuntu because the dev gave like 4 warnings even before attempting to set it up), and then I spent a whole lot of time getting used to nvim and customizing it by inlcuding plugins and trying to understand what the plugin lua scripts did. Also got to know LazyVim ≠ lazy.nvim

I felt I needed to write kernels with some context rather than isolated kernels where I’m only writing suboptimal ones and not optimizing it to the fullest, I’m currently writing kernels that get me in the top 20% or so on the tensara leaderboards but going from a naive slow to a 2x speed up is easy, getting those minute increments is what pushes you to the top and where it gets hard.

So I decided to work on mnist in cuda

![I like to sketch a brief outline before I start working](blog/cuda/CUDA/image%202.png)

I like to sketch a brief outline before I start working

### MNIST in CPP

Part 1 is simply cpp, just to get a hand of things, despite the sketch we ditched the layer class because we were writing only a small network with two layers, and used symbolic differentation. Due to the fact that we use symbolic differentiation and no autograd, it becomes quite tedious for large networks and hence scalability is out of question

Code - 

[mnist-cuda/cpp at main · grasgor/mnist-cuda](https://github.com/grasgor/mnist-cuda/tree/main/cpp)

Also libtorch was a pain to work with (even though I used it just for dataloading)

# Day 10

I hadnt documented the last 4 days because well, I wasn’t reading the PMPP book as I told myself nor did I read the modal glossary. NVIDIA has now come up with a Blackwell hackathon that starts 10th Nov (today). 

We gotta read on NVFP4, find out what the hype is about.

I found out an accelerated computing course that I’d like to do (mainly because it focuses on the broader goal of understanding hardware acceleration through excercises like writing the mandelbrot.

[https://x.com/grasgor/status/1987722936791519280?s=20](https://x.com/grasgor/status/1987722936791519280?s=20)

NVIDIA also has a new set of videos I’d like to watch - https://www.youtube.com/watch?v=Sdjn9FOkhnA&t=5s

Twitter timeline is flooded with people taking up the “100 days GPU grind”, while I did take it up as well, it has increased quite a bit now, and I almost sometimes find it (not useful) to give day updates even though you just wrote the “__global__” keyword and called it a day. But then again no accounting no credibility, you slack of, so better something than nothing. 

## Leaky Relu

~560 GFLOPS on B200, 10th on the leaderboard

back to tensara

```cpp
#include <cuda_runtime.h>
__global__ void leaky_relu(const float* input, float alpha, float* output, size_t n, size_t m){

    float4* input4 = (float4*)input;
    float4* output4 = (float4*)output;
    const uint threadId = blockDim.x * blockIdx.x + threadIdx.x;
    const uint totalThreads = n * m / 4;

    if (threadId < totalThreads) {
        float4 val = input4[threadId];
        float4 alpha_val = make_float4(alpha * val.x, alpha * val.y, alpha * val.z, alpha * val.w);
        val.x = fmaxf(alpha_val.x, val.x);
        val.y = fmaxf(alpha_val.y, val.y);
        val.z = fmaxf(alpha_val.z, val.z);
        val.w = fmaxf(alpha_val.w, val.w);
        output4[threadId] = val;
    }
}

__global__ void leaky_relu_2D_strided(const float* input, float alpha, float* output, size_t n, size_t m) {
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= n) return;

    const size_t m4 = m / 4;      // number of full float4 blocks
    const size_t tail = m % 4;    // leftover floats per row

    const size_t x4_start = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;

    // Process full float4 blocks
    for (size_t x4 = x4_start; x4 < m4; x4 += stride) {
        size_t idx = y * m + x4 * 4;
        float4 val = reinterpret_cast<const float4*>(input)[y * m4 + x4];
        val.x = fmaxf(alpha * val.x, val.x);
        val.y = fmaxf(alpha * val.y, val.y);
        val.z = fmaxf(alpha * val.z, val.z);
        val.w = fmaxf(alpha * val.w, val.w);
        reinterpret_cast<float4*>(output)[y * m4 + x4] = val;
    }

    // Handle leftover floats (tail)
    if (tail > 0 && x4_start == 0) {  // only one thread handles the tail per row
        size_t start = m4 * 4;
        for (size_t x = start; x < m; ++x) {
            size_t idx = y * m + x;
            output[idx] = fmaxf(alpha * input[idx], input[idx]);
        }
    }
}

__global__ void leaky_relu_2D(const float* input, float alpha, float* output, size_t n, size_t m) {
    float4* input4 = (float4*)input;
    float4* output4 = (float4*)output;
    
    const size_t x4 = blockIdx.x * blockDim.x + threadIdx.x;  
    const size_t y  = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= n) return;

    const size_t m4 = m / 4;       
    const size_t tail = m % 4;     

    if (x4 < m4) {
        float4 val = input4[y * m4 + x4];
        float4 alpha_val = make_float4(alpha * val.x, alpha * val.y, alpha * val.z, alpha * val.w);
        val.x = fmaxf(alpha_val.x, val.x);
        val.y = fmaxf(alpha_val.y, val.y);
        val.z = fmaxf(alpha_val.z, val.z);
        val.w = fmaxf(alpha_val.w, val.w);
        output4[y * m4 + x4] = val;
    }

    // Handle leftover
    if (tail > 0 && x4 == 0) {
        const size_t start = m4 * 4;
        for (size_t x = start; x < m; ++x) {
            const size_t idx = y * m + x;
            const float v = input[idx];
            output[idx] = fmaxf(alpha * v, v);
        }
    }
}

__global__ void leaky_relu_1D_stride(const float* input, float alpha, float* output, size_t n, size_t m) {
    const size_t total_elements = n * m;
    const size_t total_float4 = total_elements / 4;
    const size_t tail = total_elements % 4;

    const size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x; // stride in float4 units

    // Process full float4 blocks
    for (size_t i = threadId; i < total_float4; i += stride) {
        float4 val = reinterpret_cast<const float4*>(input)[i];
        val.x = fmaxf(alpha * val.x, val.x);
        val.y = fmaxf(alpha * val.y, val.y);
        val.z = fmaxf(alpha * val.z, val.z);
        val.w = fmaxf(alpha * val.w, val.w);
        reinterpret_cast<float4*>(output)[i] = val;
    }

    // Handle leftover floats (tail) by thread 0
    if (threadId == 0 && tail > 0) {
        size_t start = total_float4 * 4;
        for (size_t i = start; i < total_elements; ++i) {
            output[i] = fmaxf(alpha * input[i], input[i]);
        }
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {

    // int threadsPerBlock = 1024;
    // int numBlocks = (n*m/4 + threadsPerBlock - 1) / threadsPerBlock;
    // leaky_relu<<<numBlocks, threadsPerBlock>>>(input, alpha, output, n, m);

    dim3 threadsPerBlock(32,32,1);
    dim3 blocksPerGrid(((m + 3)/4 + threadsPerBlock.x - 1)/threadsPerBlock.x, (n + threadsPerBlock.y - 1)/threadsPerBlock.y);
    leaky_relu_2D<<<blocksPerGrid, threadsPerBlock>>>(input, alpha, output, n, m);
    // leaky_relu_2D_strided<<<blocksPerGrid, threadsPerBlock>>>(input, alpha, output, n, m);
    // leaky_relu_1D_stride<<<numBlocks, threadsPerBlock>>>(input, alpha, output, n, m);
}
```

I wrote a bunch of kernels, some to clear out confusion and test some hypothesis. So strided kernels are useful when you’re launching less threads or cant launch enough threads but otherwise they slow you down due to non coalesced memory access.

The second thought I had was, when the matrix is ultimately flattened in memory, what is the point of using a 2D kernel and thread indices just because a mental model thinks of matrices in 2D. Although I expected the 2D kernel to work the same as a 1D kernel, the 1D one was ~20 GFLOPS faster which I cant figure out why yet.

![image.png](blog/cuda/CUDA/image%203.png)

## Image Thresholding

~2676 GFLOPS on B200, 2nd on leaderboard

```cpp
#include <cuda_runtime.h>

__global__ void threshold(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    const float4* input4 = reinterpret_cast<const float4*>(input_image);
    float4* output4 = reinterpret_cast<float4*>(output_image);

    size_t num_elements = height * width;
    size_t head = num_elements / 4;
    size_t tail = num_elements % 4;

    if (x < head) {
        float4 val = input4[x];
        val.x = (val.x > threshold_value) ? 255.0f : 0.0f;
        val.y = (val.y > threshold_value) ? 255.0f : 0.0f;
        val.z = (val.z > threshold_value) ? 255.0f : 0.0f;
        val.w = (val.w > threshold_value) ? 255.0f : 0.0f;
        output4[x] = val;
    }

    if (tail > 0 && x == 0) {
        const size_t start = head * 4;
        for (size_t i = start; i < num_elements; ++i) {
            output_image[i] = (input_image[i] > threshold_value) ? 255.0f : 0.0f;
        }
    }
}

extern "C" void solution(const float* input_image, float threshold_value, float* output_image, size_t height, size_t width) {
    size_t num_pixels = height * width;
    size_t num_vec4 = num_pixels / 4;

    int threadsPerBlock = 1024;
    int numBlocks = (num_vec4 + threadsPerBlock - 1) / threadsPerBlock;

    threshold<<<numBlocks, threadsPerBlock>>>(input_image, threshold_value, output_image, height, width);
}

```

float4 vectorized memory access whenever possible is now a standard of mine and is almost guaranteed to give speed ups, you just have to make sure you handle the tail elements ie the elements that may be missed out because total_elements % 4 ≠ 0

## Misc

The accelerated computing lab 1 requires me to read up and learn about SIMD and AVX instruction sets and the internals to speed up the Mandelbrot code by vectorizing it on CPU, this would serve as a precursor to better understanding vectorization on the GPU later on.

CUDA MNIST is in progress, initially I was writing individual kernels for each function ie matmul, relu, bias addition etc, but now I figured we should rather write fused kernels, so I decided to write both version and see purely for the sake of it how they differ in performance.

Also figured we should be pre allocating everything including space needed for gradients on memory to avoid the memory transfer overhead in between kernel calls, and even in frameworks like pytorch pre allocation is often done (and can easily be calculated once we know input output dim specified by the user and dimensions of hidden layers) except when the forward calls involve conditionals.

### DeepML

Tried doing this but I ran out of submissions because I spammed the button despite typos in my code.

[https://x.com/grasgor/status/1987902005738922015?s=20](https://x.com/grasgor/status/1987902005738922015?s=20)

# Day 11

A lot of miscellaneous things, and also was quite surprised seeing the response on my tweet, coicidentally I slacked the very next day, but I’ll still count it, I do not have a bar I have set myself for what should be counted but for the sake of continuity, giving my self a little longer leash to get started on a streak.

Made a submission to the NVFP4 Hack, mainly to get familiar with things, I used the default [submission.py](http://submission.py) itself, funnily enough it is pretty performant and stands right in the middle of the leaderboard serving as a good baseline to continue from there on 

## Fused GEMM

Since a part of my MNIST project requires me to implement fused kernels (a self imposed thing), I decided to write a performant kernel on Tensara first.

- 526GFLOPS on B200
- 488GFLOPS on H100
- 61GFLOPS on T4

```cpp
#include <cuda_runtime.h>

__global__ void fused_gemm(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M){

    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x; //columns
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y; //rows

    // matmul part
    if(x < M && y < B){
            float temp = 0.0f;
            for(int k = 0; k < N; k++){

                temp = fmaf(A[y*N + k], W[x*N + k], temp); //fused multiply and add hardware native instruction
            }
            temp += b[x]; //include bias addition
            C[y*M + x] = fmaxf(0.0f, temp); // ReLU
        }  
}

// Note: A, W, b, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* W, const float* b, float* C, size_t B, size_t N, size_t M) {

    dim3 numThreads(32, 32, 1);
    dim3 numBlocks((M + numThreads.x - 1) / numThreads.x,
                   (B + numThreads.y - 1) / numThreads.y);

    fused_gemm<<<numBlocks, numThreads>>>(A, W, b, C, B, N, M);
}
```

We are far off the top, this is one of the first kernels where I have felt that we are barely scratching the surface. We have to reach ~40TFLOPS, thats around an 80x speed up

## Notes on SIMD

Stands for single instruction multiple data (Still need to read)

# Day 12

We’re reading the following posts in order to understand shared memory

[https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)

https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/

[https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)

[https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)

### Tip 1

Run  even the smallest kernels on gpu because that ensures intra device memory transfers that has high mem bandwidth as opposed to host → device

If you need to run host → device transfer, use pagelock

Prefere larger chunks rather than batches

inter device transfers can be overlapped