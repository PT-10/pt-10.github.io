---
id: compute-vs-memory
title: Are You Compute-Bound or Memory-Bound?
date: Nov 23, 2025
snippet: Understanding the bottleneck of your CUDA kernels.
tags: [cuda, gpu, performance]
---


> The key to optimizing your executions, in this case we'll be talking about CUDA kernels (you can consider any function accelerator to that matter) is understanding whether you are compute-bound or memory-bound.
> 

# Intuition

In plain English, being **memory-bound** means that your **execution speed is capped by the speed of data transfers** i.e **memory bandwidth**.

**Compute-bound** means that the **operations cannot be performed any faster** i.e **performance is limited by how many operations the hardware can execute per second**.

## The postman example

Consider a small neighbourhood post office, a couple of kilometres away from your home that operates manually. There are 4 postmen at the office. Each one of them caters to a particular area and can deliver a maximum of 20 mails at once.

### **Memory-Bound Kernel**

![image.png](/blog/compute-vs-memory/image.png)

On a certain day, the post office happened to receive only 20 mails. Even though their sorting capacity was 80, the sorting staff were idle part of the time, waiting for mail.

Here, the bottleneck is how fast letters arrive. Improving compute power i.e their sorting capacity does not improve their throughput of 20 mails being delivered. So the kernel is memory-bound.

In terms of CUDA kernels, the GPU can perform many more FLOPs, but it spends time **waiting for data** to arrive from global memory. This would mean that to improve throughput we would need to increase memory bandwidth either by loading lower precision data, caching memory for faster transfer speeds, improving read access patterns to reduce the number of data reads.

### **Compute-Bound Kernel**

![image.png](/blog/compute-vs-memory/image-1.png)

During the holidays, the post office may receive an upwards of 100 mails. However, the office is able to sort only 80 mails. 

That means the delivery side (memory bandwidth) is **fast enough**. The sorting staff (compute units) are working at **full capacity**. Even if delivery becomes faster, the system will still only finish **80 letters at most**, because the sorter is the limiting step. So this kernel is compute-bound.

In terms of CUDA kernels, this would mean that even if the GPU could fetch more data from memory, the throughput would not increase because the cores are already operating at max capacity.

# Roofline Model

![image.png](/blog/compute-vs-memory/image-2.png)
