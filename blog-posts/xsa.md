---
id: xsa
title: Exclusive Self Attention
date: Apr 26, 2026
snippet: Is this the upgrade self attention needs?
tags: [attention, deep-learning, llm]
---

> I'll jump straight to the formulation and work from there to establish intuitions for what XSA means compared to vanilla self-attention


# Self-Attention

Standard (causal) self-attention is defined as [[1]](#/blog/xsa#eq-1):

$$
q_i = W_q x_i \in \mathbb{R}^d, \quad k_j = W_k x_j \in \mathbb{R}^d, \quad v_j = W_v x_j \in \mathbb{R}^d
$$

$$
a_{i,j} = \frac{\exp(q_i^\top k_j)}{\sum_{j'=1}^{i} \exp(q_i^\top k_{j'})}, \quad \text{with } \sum_{j=1}^{i} a_{ij} = 1
$$

<a id="eq-1"></a>
$$
\quad y_i = \sum_{j=1}^{i} a_{i,j} \, v_j \in \mathbb{R}^d
$$
<div style="text-align:right;font-size:0.8rem;color:#888;margin-top:-0.6rem;">[1]</div>

Self-attention essentially updates each token **row-wise fixing ${i}$** by taking a weighted sum of all tokens’ value vectors ${v_j \in \mathbb{R}^d}$, where the weights reflect how relevant each token is to it; since this is a weighted sum of vectors in ${\mathbb{R}^d}$, the output ${y_i}$ also lies in ${\mathbb{R}^d}$.

# Understanding the inherent bias 
Before we proceed into the core idea, let us rearrange terms in our attention output score.

$$
\quad y_i = \underbrace{a_{ii} \, v_i}_{\in \mathbb{R}^d \text{ (self contribution)}} \;+\; \underbrace{\sum_{j \ne i} a_{ij} \, v_j}_{\in \mathbb{R}^d \text{ (context from other tokens)}}
$$

Now, let's simplify this further since ${v_j \in \mathbb{R}^d}$ are ${d}$-dimensional vectors, a linear combination will also be a ${d}$-dimensional vector in the same space and rewrite it as:

$$
\quad y_i = a_{ii} \, v_i \;+\; c_i, \quad \text{where } c_i = \sum_{j \ne i} a_{ij} \, v_j \in \mathbb{R}^d
$$

## Attention similarity bias
The paper states:
> The output of attention tends to have a high cosine similarity with the self value vector

We will now work our way to show the direct dependency of the cosine similarity between the two entities on the magnitude of the self-value vector. 

Let us take the cosine similarity of attention output ${y_i}$ and the value vector ${v_i}$ corresponding to the token ${i}$:

$$
\cos(y_i, v_i) = \frac{y_i^\top v_i}{\|y_i\| \, \|v_i\|}
$$

Now substitute ${y_i = a_{ii} v_i + c_i}$:

$$
\cos(y_i, v_i) = \frac{(a_{ii} v_i + c_i)^\top v_i}{\|a_{ii} v_i + c_i\| \, \|v_i\|}
$$

$$
= \frac{a_{ii} \, v_i^\top v_i + c_i^\top v_i}{\|a_{ii} v_i + c_i\| \, \|v_i\|}
$$

To make the dependency clearer, let us decompose the context vector ${c_i}$ into it's components,  one parallel to ${v_i}$ and one orthogonal to ${v_i}$:

$$
c_i = \alpha_i v_i + r_i,
\quad
\text{where }
\alpha_i = \frac{c_i^\top v_i}{\|v_i\|^2},
\quad
r_i \perp v_i
$$

Substituting this into ${y_i}$:

$$
y_i = a_{ii}v_i + c_i
$$
$$
= a_{ii}v_i + \alpha_i v_i + r_i
$$

$$
= (a_{ii} + \alpha_i)v_i + r_i
$$

For the sake of brevity, let us re write the scalar coefficient:

$$
\beta_i = a_{ii} + \alpha_i
$$

Then, the attention output can be represented as a component along self-value vector and a residual orthognal to it [[2]](#/blog/xsa#eq-2):

<a id="eq-2"></a>
$$
y_i = \beta_i v_i + r_i,
\quad r_i \perp v_i
$$
<div style="text-align:right;font-size:0.8rem;color:#888;margin-top:-0.6rem;">[2]</div>

Now the cosine similarity becomes:

$$
\cos(y_i, v_i)
=
\frac{(\beta_i v_i + r_i)^\top v_i}
{\|\beta_i v_i + r_i\| \, \|v_i\|}
$$

Since ${r_i \perp v_i}$, we have ${r_i^\top v_i = 0}$, so:

$$
\cos(y_i, v_i)
=
\frac{\beta_i \|v_i\|^2}
{\sqrt{\beta_i^2 \|v_i\|^2 + \|r_i\|^2} \, \|v_i\|}
$$

Therefore:

$$
\cos(y_i, v_i)
=
\frac{\beta_i \|v_i\|}
{\sqrt{\beta_i^2 \|v_i\|^2 + \|r_i\|^2}}
$$

## Understanding the geometry
Here, it is important to understand the geometry of $\beta_i$ and $r_i$. If the other value vectors $v_j$, which we can call context vectors are oriented similar to the self-value vector $v_i$, then $\beta_i$ increases and the orthogonal $r_i$ decreases and vice-versa.

<div style="display:flex;flex-direction:column;gap:1rem;margin:1.5rem 0">
  <div style="text-align:center">
    <img src="/blog/xsa/seed-balanced-compressed.gif" alt="Balanced context vectors" style="width:100%;border-radius:6px"/>
    <p style="font-size:0.8rem;margin-top:0.4rem;color:#888">Balanced — moderate β and r</p>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
    <div style="text-align:center">
      <img src="/blog/xsa/seed-aligned-compressed.gif" alt="Aligned context vectors" style="width:100%;border-radius:6px"/>
      <p style="font-size:0.8rem;margin-top:0.4rem;color:#888">Strong alignment — large β, small r</p>
    </div>
    <div style="text-align:center">
      <img src="/blog/xsa/seed-perpendicular-compressed.gif" alt="Perpendicular context vectors" style="width:100%;border-radius:6px"/>
      <p style="font-size:0.8rem;margin-top:0.4rem;color:#888">Perpendicular — small β, large r</p>
    </div>
  </div>
</div>

So far we have worked our way through the math to understand that there is going to be some bias i.e cosine similarity between the attention output and the self-value vector.

## Why is this cosine similarity high in trained LLMs?

Let us look at the numerator of the cosine similarity in the non expanded form in [[3]](#/blog/xsa#eq-3).

<a id="eq-3"></a>
$$
y_i^\top v_i = a_{ii} \|v_i\|^2 + \sum_{j \ne i} a_{ij}(v_j^\top v_i)
$$
<div style="text-align:right;font-size:0.8rem;color:#888;margin-top:-0.6rem;">[3]</div>


The first term $a_{ii} \|v_i\|^2$ is always positive since all the coefficients $a_{ij}$ are a result of softmax where $\sum a_{ij} = 1$.
In the seond term $\sum_{j \ne i} a_{ij}(v_j^\top v_i)$, while the coefficients are always positive the dot product $(v_j^\top v_i)$ has also been empirically observed to be positively correlated.

Several works support this:

- **[Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)** *(Abnar & Zuidema, ACL)*  

- **[On the Expressive Power of Self-Attention Matrices](https://arxiv.org/abs/2106.03764)** *(Likhosherstov et al., AAAI)*  

- **[Representation Degeneration Problem in Training Natural Language Generation Models](https://arxiv.org/abs/1907.12009)** *(Jun et al., ICLR)*  


Putting these together:

$$
\cos(y_i, v_i) \gg 0
$$

As depth increases, tokens attend more to themselves i.e $a_{ii} \gg \sum_{j \ne i} a_{ij}$ and thus the bias should generally increase with depth.

# Exclusive Self-Attention (XSA) 

From the earlier decomposition in [[2]](#/blog/xsa#eq-2), we already have attention output as a self-aligned component plus an orthogonal residual.

## The Idea
In a transformer:
- the self-attention mechanism is responsible for context aggregation (mixing token information)
- the FFN layer is responsible for point-wise feature transformation 

The hypothesis is that, in vanilla self-attention, the attention mechanism ends up doing both of these. This is empirically shown using the high attention-similarity bias across layers.
> "The prevalence of the attention similarity bias suggests that SA spends a significant portion of its capacity modeling the point wise feature transformation."

<figure style="text-align:center; margin:1rem 0;">
  <img src="/blog/xsa/image.png" alt="XSA" style="max-width:100%; border-radius:6px;" />
  <figcaption style="font-size:0.85rem; color:#888; margin-top:0.4rem;">
    Attention-similarity bias in a 1.3B LM (sequence length 2048), trained on 100B tokens and averaged over 1,024 random training sequences: left, mean cosine similarity cos(<i>v</i><sub>i</sub>, <i>v</i><sub>j</sub>); middle, mean diagonal attention <i>a</i><sub>i,i</sub>; right, mean cosine similarity cos(<i>y</i><sub>i</sub>, <i>v</i><sub>i</sub>).
  </figcaption>
</figure>


The paper defines XSA in [[4]](#/blog/xsa#eq-4) as follows (with the first line same as standard SA in [[1]](#/blog/xsa#eq-1)):

<a id="eq-4"></a>
>
>$$
> \quad y_i = \sum_{j=1}^{i} a_{i,j} v_j
>$$
>
>$$
>z_i = y_i - \frac{y_i^\top v_i}{\|v_i\|^2} \, v_i
>$$
<div style="text-align:right;font-size:0.8rem;color:#888;margin-top:-0.6rem;">[4]</div>


XSA introduces an additional step that removes the projection of the attention output $y_i$ onto the self-value vector $v_i$.

From [[2]](#/blog/xsa#eq-2) we had:

$$
y_i = \beta_i v_i + r_i, \quad r_i \perp v_i
$$

Following the XSA step:
Let us substitute the decomposition of $y_i$:

$$
z_i = (\beta_i v_i + r_i) - \frac{(\beta_i v_i + r_i)^\top v_i}{\|v_i\|^2} \, v_i
$$

Since $r_i \perp v_i$, we have $r_i^\top v_i = 0$, so:

$$
y_i^\top v_i = \beta_i \|v_i\|^2
$$

Substituting back:

$$
z_i = (\beta_i v_i + r_i) - \frac{\beta_i \|v_i\|^2}{\|v_i\|^2} v_i
$$

$$
= (\beta_i v_i + r_i) - \beta_i v_i
$$

<a id="eq-5"></a>
$$
= r_i
$$
<div style="text-align:right;font-size:0.8rem;color:#888;margin-top:-0.6rem;">[5]</div>


As a result:
- $z_i$ no longer contains $v_i$ itself  
- nor any component from the context that is aligned with $v_i$


Since [[5]](#/blog/xsa#eq-5) and $r_i \perp v_i$, we have:

$$
z_i^\top v_i = 0
$$

Therefore:

$$
\cos(z_i, v_i) = 0
$$

By removing the component along $v_i$, XSA enforces a cleaner separation:
- the self-attention mechanism learns *new* information from tokens other than itself
- the FFN layer performs the point-wise feature transformations

It is also hypothesized that self-information is not lost by doing so because self-information already has a direct path through residual connections.  


> Instead of mixing everything together, XSA forces attention to contribute only what is **new**.
