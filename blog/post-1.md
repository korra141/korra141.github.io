---
layout: post
title: "Rigid Body Motions"
date: 2026-05-18
---

# Rigid Body Motions

*A short introduction to the math behind how robots move in 3D space.*

---

## What is a Rigid Body?

*Define rigid body — distances between points are preserved under motion. Contrast with deformable bodies.*

## Representing Rotation

*SO(2) for 2D, SO(3) for 3D. Rotation matrices, properties (orthogonal, determinant = 1).*

```
R ∈ SO(3)  ⟺  Rᵀ R = I,  det(R) = 1
```

## Representing Pose: SE(3)

*Combining rotation and translation into a single transformation. Homogeneous coordinates.*

```
T = | R  t |  ∈ SE(3)
    | 0  1 |
```

## Why Lie Groups?

*Rotations don't live in Euclidean space — naive addition doesn't stay on the manifold. Lie groups give the right structure for composition, interpolation, and differentiation.*

## Practical Implications

*How this shows up in SLAM, state estimation, and manipulation — anywhere a robot needs to reason about where it is and how it's oriented in space.*

---

*Work in progress. Part of a series on the math behind robotics.*
