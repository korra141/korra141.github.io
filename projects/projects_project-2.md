---
layout: page
title: "SLAM in Degenerate Feature Environments"
---

# SLAM in Degenerate Feature Environments

**Repo:** [github.com/korra141/slam-fusion](https://github.com/korra141/slam-fusion) \
**Stack:** ROS2 Humble · LiDAR · IMU · RGB-D · LIO-SAM · Super Odometry \
**Status:** In progress

---

> State-of-the-art LiDAR-inertial odometry performs well in feature-rich indoor environments but degrades significantly in long, featureless corridors and low-texture outdoor terrain, where geometric constraints are too sparse for reliable point cloud registration. This project investigates whether multimodal sensor fusion — adding visual and inertial cues to LiDAR — can recover robustness where geometry alone fails, and at what point environmental sparsity breaks odometry entirely.

---

## The Gap

Geometric SLAM relies on point cloud registration finding enough distinct structure to constrain a pose estimate. In long, self-similar corridors or sparse outdoor terrain, that structure disappears — registration underconstrains the solution and drift accumulates without bound. Loop closure is the usual fix, but in repetitive, self-similar spaces it is also where loop closure is most likely to produce false positives, correcting drift with the wrong correction.

## Research Questions

1. Does loop closure genuinely correct drift in degenerate environments, or does it mask an underlying sensor limitation that resurfaces elsewhere?
2. Can multimodal fusion — adding visual features to LiDAR-inertial odometry — reduce drift without relying on loop closure at all?
3. How does tight versus loose sensor coupling affect robustness specifically in degenerate, feature-sparse scenarios?
4. At what threshold of environmental sparsity does odometry fail outright, regardless of sensor suite?

## Hypotheses

LiDAR-inertial systems alone are expected to accumulate drift in corridors due to geometric degeneracy — there simply isn't enough structure to constrain registration. Adding visual features should reduce that drift by supplying constraints LiDAR can't. Tightly coupled sensor fusion should outperform loosely coupled approaches under degeneracy, since it can weight and correct between modalities at each step rather than after the fact. And loop closure, rather than being a universal fix, may generate false positives in self-similar spaces — a risk that needs to be measured, not assumed away.

## Dataset & Experiments

Data is collected in long indoor corridors using LiDAR, IMU, and (planned) camera input. Six experiments compare Super Odometry, KISS-ICP, FAST-LIO2, VINS-Mono, and LVI-SAM against each other, with and without loop closure enabled, to isolate what loop closure is actually contributing in these environments versus what the sensor fusion itself provides.

## Evaluation Methodology

Ground truth trajectories aren't available for these environments, so evaluation relies on proxies: ICP fitness score between submaps, and map consistency across multiple passes through the same corridor (MME). A method that drifts will disagree with itself on repeated passes even without ground truth to compare against.

## What's Next

Extending the current LiDAR-inertial backbone (built on LIO-SAM and Super Odometry) with visual-inertial odometry on ROS2 Humble, and building out the localisation uncertainty model — propagating uncertainty under uninformative observations rather than assuming a fixed noise level, which connects directly to the non-Gaussian, Lie-group uncertainty estimation developed in the thesis work.
