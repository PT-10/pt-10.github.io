---
id: flash-attention
title: Flash Attention
date: Apr 11, 2026
snippet: Implementing flash attention
tags: [cuda, gpu, performance]
---


> Attention was all you needed, but attention needed someone else. There came Tri Dao to the rescue, bringing in flash to help attention. (I suppose I made it too cheesy)


![Flash](/blog/flash-attention/image.png)



 # Background

 ## Memory Hierarchy

Just like a CPU has L1/L2/L3 cache sitting between the processor and RAM, GPUs have a similar hierarchy. From fastest to slowest:

- **Registers / SRAM (shared memory)** — tiny, blazing fast, lives on-chip inside each SM (streaming multiprocessor)
- **HBM (High Bandwidth Memory)** — the GPU's main memory, large but much slower than SRAM

The golden rule: the closer data is to the compute units, the faster the kernel runs.

 ## Forward pass and kernel calls

When you do:

```python
device = torch.device('cuda')
model = LLM().to(device)
```

All model weights — $W_Q, W_K, W_V, W_O$, MLP weights, LayerNorm parameters — get loaded into HBM. That's your GPU's "RAM". Every kernel that runs during the forward pass has to pull data from there.

Each kernel follows the same pattern:

> HBM → SRAM → compute → HBM

Load inputs from HBM into fast on-chip SRAM, do the math, write results back to HBM. The next kernel then reads those results from HBM again.

## Why vanilla attention is slow

Self-attention computes:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Let's break down what actually happens on the GPU, step by step:

```python
scores = Q @ K.T          # [N, N] matrix
scores = scores / sqrt(d)
probs  = softmax(scores)
out    = probs @ V
```

Each of these lines is a **separate kernel**. And each kernel follows that read-compute-write pattern. So for a sequence length of $N = 16384$ with float16:

$$N^2 \times 2 \text{ bytes} = 16384^2 \times 2 \approx 512 \text{ MB per head}$$

That's half a gigabyte of attention scores materialised in HBM — and it gets read and written multiple times across the four kernels. As we saw in the [compute vs memory post](#/blog/compute-vs-memory), attention is firmly [memory bound](#/blog/compute-vs-memory): the GPU cores sit waiting while data shuttles back and forth to HBM.

## The fix is obvious (in hindsight)

If the problem is repeated HBM round-trips, the solution is to avoid them — fuse all four steps into a single kernel that keeps the intermediate results in SRAM and never writes them to HBM.

That's exactly what FlashAttention does.

 # Attention!
 Vanilla attention is [memory bound](#/blog/compute-vs-memory). Here's how -\
 Consider a model with context length 16k i.e the input $Q$. $K$, $V$ matrices are of shape $(N, d)$ where N = 16,384. With self-attention being calculated as 
 $O = softmax(\frac{QK^T}{V})V$