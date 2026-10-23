---
id: diffusion-from-first-principles
title: Diffusion from first principles
date: April 23, 2026
snippet: Intuition behind generative models and how we ended up with diffusion
tags: [image-generation, deep-learning]
---

# Data as a probability distribution

Look around, you must have seen various breeds of dogs. They're all called dogs, but have differences in features like their size, colour, furriness that make them vary from each other.

> In machine learning it is common to realize this diversity of possible images as a probability distribution over the space of images. We call such a distribution a data distribution and denote it as $p_{data}$

## Statistical Perspective for Real Images

Let us consider RGB images with a resolution of 256×256. An image can be represented by a 256×256×3 vector. This means that an image is a point in a 256×256×3-dimensional space. Now, not every pixel combination would correspond to a coherent image. Even when we consider a broad set of dog images, there would only be certain combinations of pixels that make the point represent a dog.

Without loss of generality, let us consider the image ${x}$ to now be a point in a ${d}$-dimensional space:\
The data distribution $p_{data}$ can be understood as a probability density function $\mathbb{R}^d \rightarrow \mathbb{R}_{\ge 0}$ that assigns each object $x$ a likelihood $p \sim {data}(x) \ge 0$.

![Butterfly SketchRNN](/blog/diffusion-from-first-principles/pixelspace.png)

# Generation as sampling

Suppose you are with your friends, and you ask everybody to sketch a butterfly.

![Butterfly SketchRNN](/blog/diffusion-from-first-principles/image-dark.png)

You notice something, all the sketches have semblance to what you'd call a butterfly. There is no single correct sketch, unless somebody went ahead and drew a bird that they're calling a butterfly. Everyone has seen a butterfly before; either in photos, cartoons, videos or real life. They vary is wing shapes, colors etc. This represents $p_{data}$.

Now creating a sketch ie generating an object ${x}$ can be considered as sampling from the data distribution ${x} \sim p_{data}$.

***

# Modelling the data distribution

We have established that the real data can be understood as a probability density function that assigns likelihood to images in the dataset. We have also established that generating a new image is modeled as sampling from the data distribution. Thus, if we know the probability density function $p_{data}$, we can directly sample from it to generate new images.

## Modelling $p_{data}$ directly is hard

Consider a binary image where each pixel can take values ${0,1}$. For an image of resolution $H \times W$, we would have $d = H \cdot W$, and a total of $2^d$ possible vectors.

For a $256 \times 256$ binary image, this gives $d = 65,536$ i.e, $2^{65,536} \approx 10^{19,728}$ possible vectors. Modelling $p_{data}$ directly would require assigning a probability to each of these configurations, which is computationally and memory-wise impossible.

Real images are even more complex: a $256 \times 256$ RGB image has $196,608$ interleaved pixels with 256 possible values for each, leading to $256^{196,608} = 2^{1,572,864}$ possible vectors. Besides, only a tiny fraction of these correspond to meaningful images. Thus, in practice $p_{data}$ is extremely complex. It lives in a very high-dimensional space, and assigning exact likelihoods to images is intractable.

## The Basic Idea

Instead of modelling $p_{data}(x)$ directly, we map a distribution $p(z)$ to our data distribution. Then, the steps to generate data would look like:
<img src="/blog/diffusion-from-first-principles/mapping_dark.png" style="float:right; width:40%; margin:0 0 1rem 2rem;" />

1. Sample a latent variable $z \sim p(z)$

2. Map it to a data point using a function ${x = f(z)}$

<div style="clear:both"></div>

The next question is - how do we find out this mapping function? What does our $p(z)$ look like? We'll train a neural network to learn that!

# Autoencoders and VAE (Variational Autoencoder)
If we had two sets $A$ and $B$, where:
- $A$ is the set of real data points  
- $B$ is the set of latent points that map to valid data

then learning the mapping would be straightforward. We could simply learn a function: $f: B \rightarrow A$ and use it to generate new samples by picking $z \in B$ and computing $x = f(z)$.

The problem is we don’t know what $B$ is. So how do we guarantee that a latent is mapped to a data 
point of the data distribution? 

## Autoencoders
<img src="/blog/diffusion-from-first-principles/autoencoder.png" style="display:block; margin:0 auto; width:70%;" />

We learn an encoder that maps data to the latent, and then to ensure a valid mapping, the decoder is trained to reconstruct the input image from the mapped latent. This is called an **autoencoder**.

However, this has a drawback from the perspective of generation. Whilst we have successfully learnt the mapping from the latent space to the data distribution, we do not know how $z$ is distributed. Because of this, we cannot sample a new point $z \sim p(z)$. If we can't sample, we can't generate a new image $x$.

## VAE
To generate new images, we need to be able to sample from the latent space. A Gaussian distribution is simple and easy to sample from. So if $z \sim p(z) = \mathcal{N}(0, I)$, then we can sample a point from the latent and use the mapping that the decoder has learnt to generate a new image.

The math here relies on two tools — [MLE](#/blog/probability-refresher#maximum-likelihood-estimation) and [KL divergence](#/blog/probability-refresher#kl-divergence) — covered in the [probability refresher](#/blog/probability-refresher) if you need a recap.


### Variational Inference

#### $p(x)$ as an integral
Recollect that our goal was to model $x \sim p_{data}$. For the sake of notation let us write $p_{data} = p_\theta(x)$ where $\theta$ are the parameters that govern the underlying probability density function. Since it was hard to model it directly, we introduced a latent variable $z \sim p(z)$.

From probability theory, we can rewrite $p_\theta(x)$ as the sum of marginals $p_\theta(x|z)$ over all $z$. Therefore:

$$
p_\theta(x) = \sum_{z} p_\theta(x \mid z)\, p(z)
$$

which becomes, in a continuous case:

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz
$$
#### Motivating VAE loss via Maximum Likelihood Estimation

In MLE, we aim to find the parameters $\theta$ that maximise the likelihood of the observed data. By doing so, we identify the parameters of the pdf that best explain our data.

We define the **likelihood function** as:
$$
\mathcal{L}(\theta) = p_\theta(x)
$$

So the MLE objective becomes:
$$
\hat{\theta} = \arg\max_{\theta} \; \mathcal{L}(\theta)
$$

Using our latent variable formulation, this can be written as:
$$
\mathcal{L}(\theta) = \int p_\theta(x \mid z)\, p(z)\, dz
$$

In practice, we maximise the **log-likelihood** instead:
$$
\hat{\theta} = \arg\max_{\theta} \; \log \mathcal{L}(\theta)
$$

So our objective becomes:
$$
\log \mathcal{L}(\theta) = \log \int p_\theta(x \mid z)\, p(z)\, dz
$$

We cannot directly optimise:
$$
\log \mathcal{L}(\theta) = \log \int p_\theta(x \mid z)\, p(z)\, dz
$$

To make progress, we introduce a distribution $q_\phi(z \mid x)$ and multiply and divide inside the integral:

$$
\log \mathcal{L}(\theta) = \log \int q_\phi(z \mid x)\, \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} \, dz
$$



Recall that for a continuous random variable:
$
\mathbb{E}[f(x)] = \int f(x)\, p(x)\, dz
$

Now if we choose $q_\phi(z \mid x)$ as our distribution, we can match the same form:

$$
\int q_\phi(z \mid x)\, \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} \, dz
$$

This is exactly of the form $\int f(x)\, p(x)\, dx$, where:
$
f(x) = \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)}
$

So we can rewrite it as an expectation:
$$
= \log \mathbb{E}_{q_\phi(z \mid x)} \left[ \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} \right]
$$
Using Jensen’s inequality:
$$
\geq \mathbb{E}_{q_\phi(z \mid x)} \left[ \log \frac{p_\theta(x \mid z)\, p(z)}{q_\phi(z \mid x)} \right]
$$

We will now expand the log and rearrange the terms so that it starts looking familiar, and hopefully you can also relate it to the structure of VAE intuitively
$$
= \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z) - \log \frac{q_\phi(z \mid x)}{p(z)}]
$$

Now if you look at the second term, we can rewrite it as the KL divergence between $q_\phi(z \mid x)$ and ${p(z)}$

$$
= \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{KL}(q_\phi(z \mid x) \| p(z))
$$

Now, when I read this, the math is something that can be followed, but I want you to look at the pdfs involved and understand what the represent.

### Making sense of the loss function
When training neural nets, we usually seek to minimize the loss function. The loss function is a measure of how far our model prediction is to the true value, the smaller it i

#### Prior Distribution

In our case, $q_\phi(z \mid x)$ is what the encoder thinks the latent looks like, and $p(z) = \mathcal{N}(0, I)$ is what we want it to look like. So we minimise this difference.


