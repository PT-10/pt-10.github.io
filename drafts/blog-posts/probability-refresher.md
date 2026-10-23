---
id: probability-refresher
title: Probability Refresher
date: Apr 23, 2026
snippet: A quick refresher on MLE and KL divergence — the two tools you need to understand VAEs and diffusion models.
tags: [probability, math, deep-learning]
---

The two concepts below come up constantly in generative modelling. This is a quick, self-contained refresher.

# Maximum Likelihood Estimation

The core idea: given observed data, find the parameter $\theta$ that makes that data most probable.

Consider the following [example](https://www.probabilitycourse.com/chapter8/8_2_3_max_likelihood_estimation.php):

> I have a bag that contains 3 balls. Each ball is either red or blue, but I have no information in addition to this. Thus, the number of blue balls, call it $\theta$, might be $0, 1, 2,$ or $3$. I am allowed to choose 4 balls at random from the bag with replacement. We define the random variables $X_1, X_2, X_3, X_4$ as:
>
> $$X_i = \begin{cases} 1 & \text{if the } i\text{th chosen ball is blue} \\ 0 & \text{if the } i\text{th chosen ball is red} \end{cases}$$
>
> Note that $X_i$'s are i.i.d. and $X_i \sim \text{Bernoulli}\!\left(\frac{\theta}{3}\right)$. A Bernoulli random variable models a single binary outcome — it is 1 with probability $p$ and 0 with probability $1-p$. Here, $p = \frac{\theta}{3}$.
>
> After doing the experiment, we observe $x_1 = 1,\ x_2 = 0,\ x_3 = 1,\ x_4 = 1$ — i.e. 3 blue balls and 1 red ball.
>
> 1. For each possible value of $\theta$, find the probability of the observed sample $(x_1, x_2, x_3, x_4) = (1, 0, 1, 1)$.
> 2. For which value of $\theta$ is the probability of the observed sample the largest?

Since $X_i \sim \text{Bernoulli}\!\left(\frac{\theta}{3}\right)$, the PMF of $X_i$ is:

$$P_{X_i}(x) = \begin{cases} \dfrac{\theta}{3} & x = 1 \\[6pt] 1 - \dfrac{\theta}{3} & x = 0 \end{cases}$$

Since the $X_i$'s are independent, the joint PMF factors as:

$$P_{X_1 X_2 X_3 X_4}(x_1, x_2, x_3, x_4) = P_{X_1}(x_1)\, P_{X_2}(x_2)\, P_{X_3}(x_3)\, P_{X_4}(x_4)$$

Therefore:

$$P_{X_1 X_2 X_3 X_4}(1,0,1,1) = \frac{\theta}{3} \cdot \left(1 - \frac{\theta}{3}\right) \cdot \frac{\theta}{3} \cdot \frac{\theta}{3} = \left(\frac{\theta}{3}\right)^3\!\left(1 - \frac{\theta}{3}\right)$$

The joint PMF depends on $\theta$, so we write it as $P(\mathbf{x};\theta)$. Evaluating for each possible $\theta$:

<table style="width:auto; margin:1.5rem auto; border-collapse:collapse; text-align:center;">
  <thead>
    <tr>
      <th style="padding:0.5rem 1.5rem; border-bottom:1px solid #475569;">θ</th>
      <th style="padding:0.5rem 1.5rem; border-bottom:1px solid #475569;">P(1,0,1,1; θ)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:0.4rem 1.5rem;">0</td><td>0</td></tr>
    <tr><td style="padding:0.4rem 1.5rem;">1</td><td>0.0247</td></tr>
    <tr><td style="padding:0.4rem 1.5rem;"><strong>2</strong></td><td><strong>0.0988</strong></td></tr>
    <tr><td style="padding:0.4rem 1.5rem;">3</td><td>0</td></tr>
  </tbody>
</table>

The probability is maximised at $\theta = 2$. For this reason, we choose $\hat{\theta} = 2$ as our estimate — this is the **maximum likelihood estimate (MLE)** of $\theta$.

# KL Divergence

Imagine you have some prior belief about something, represented as a distribution $q$. You see new data and update your belief to a new distribution $p$. The question is: how much did your belief change?

**KL divergence** measures how different one distribution is from another:

$$D_{KL}(q(z) \| p(z)) = \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{p(z)} \right]$$

A few properties worth noting:
- $D_{KL}(q \| p) \geq 0$ always
- $D_{KL}(q \| p) = 0$ if and only if $q = p$
- It is **not symmetric**: $D_{KL}(q \| p) \neq D_{KL}(p \| q)$ in general

# Evidence and ELBO

The **evidence** is the probability of the observed data under the model:

$$
p_\theta(x) = \int p_\theta(x \mid z)\, p(z)\, dz
$$

This is also called the **marginal likelihood**, and our original goal (MLE) is:

$$
\max_\theta \log p_\theta(x)
$$

However, this integral is intractable. So instead, we introduce an approximate posterior $q_\phi(z \mid x)$ and derive a tractable lower bound:

$$
\log p_\theta(x)
\ge
\mathcal{L}_{\text{ELBO}}(\theta, \phi)
$$

where the **Evidence Lower Bound (ELBO)** is:

$$
\mathcal{L}_{\text{ELBO}}
=
\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)]
- D_{KL}(q_\phi(z \mid x) \| p(z))
$$

It is called a *lower bound* because:

$$
\log p_\theta(x)
=
\mathcal{L}_{\text{ELBO}}
+
D_{KL}(q_\phi(z \mid x) \| p_\theta(z \mid x))
$$

and KL divergence is always non-negative.

In practice, we maximise the ELBO (or equivalently minimise its negative), which serves as a tractable surrogate for maximising the log-likelihood.

### Gaussian Distribution

**1D (Univariate Gaussian)**\
$
\mathcal{N}(z \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z - \mu)^2}{2\sigma^2}\right)
$

**Multivariate Gaussian**\
$
\mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = 
\frac{1}{\sqrt{(2\pi)^d |\boldsymbol{\Sigma}|}} 
\exp\left(-\frac{1}{2} (\mathbf{z} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{z} - \boldsymbol{\mu}) \right)
$

where:
- $ \mathbf{z} \in \mathbb{R}^d $
- $ \boldsymbol{\mu} \in \mathbb{R}^d $ is the mean vector  
- $ \boldsymbol{\Sigma} \in \mathbb{R}^{d \times d} $ is the covariance matrix  


# Jensen's Inequality
