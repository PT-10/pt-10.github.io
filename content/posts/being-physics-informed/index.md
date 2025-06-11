+++
title = 'Being Physics Informed'
date = "2025-06-10T01:40:53+05:30"
draft = true
+++

> I've been fascinated with physics simulations, I follow F1, and I wanted to learn about PINNs. So I asked GPT to come up with a 2-day plan to equip me with the priors.

{{< toc >}}

## Day 1: Fluid Dynamics + Navier-Stokes

**Goal:** Build intuition for airflow and the Navier-Stokes equations.

**1. Basic Fluid Mechanics**  
Topics: continuity equation, Navier-Stokes (momentum), Reynolds number  
Resources:  
- [MIT OCW – Fluid Mechanics (Lectures 1–3)](https://ocw.mit.edu/courses/2-25-advanced-fluid-mechanics-fall-2013/)  
- *Introduction to Fluid Mechanics* by Fox  

**2. 2D CFD Examples**  
Demos: lid-driven cavity, flow past cylinder  
Resource:  
- [CFD Python by Lorena Barba](https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/)  
Goal: See how pressure and velocity fields evolve over time.

**3. F1-Specific Aerodynamics**  
Concepts: downforce, wake, drag, turbulent boundary layers  
Watch:  
- [Engineering Explained: F1 Aerodynamics](https://www.youtube.com/watch?v=Ls0VnMdmGJY)  
Read:  
- Technical white papers or [SAE International](https://www.sae.org/) research articles

---

## Day 2: PINNs (Physics-Informed Neural Networks)

**Goal:** Understand and apply PINNs to solve PDEs like Navier-Stokes.

**1. Intro to PINNs**  
Read: [Raissi et al. – PINNs Paper](https://arxiv.org/abs/1711.10561)  
Watch: [MathDeep – PINNs Lecture](https://www.youtube.com/watch?v=NE7BuNnS3mQ)

**2. Code Walkthrough**  
Try solving 1D/2D Poisson or heat equations  
Frameworks:  
- [DeepXDE (TensorFlow)](https://deepxde.readthedocs.io/)  
- [PINNs in PyTorch](https://github.com/maziarraissi/PINNs)

**3. Navier-Stokes + PINNs**  
Read: [Raissi – PINNs for Navier-Stokes](https://arxiv.org/abs/1711.10566)  
Try: solve flow over a square or around a cylinder using PINNs

> Practically I'd take a little more than a couple days, but I'll keep this updated.