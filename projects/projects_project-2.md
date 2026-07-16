---
layout: page
title: "SLAM in Degenerate Feature Environments"
---

# SLAM in Degenerate Feature Environments

**Repo:** [github.com/korra141/slam-fusion](https://github.com/korra141/slam-fusion) \
**Stack:** ROS2 Humble · LiDAR · IMU · RGB-D · LIO-SAM · Super Odometry \
**Status:** In progress

---

> LiDAR-inertial odometry degrades in long corridors and open, low-texture terrain — directions along which point cloud registration can't constrain a pose. Super Odometry handles this by treating the IMU as the primary propagation backbone and down-weighting degenerate sensor directions using a Hessian eigenvalue test on the scan-matching cost. Early results: it degrades gracefully when the degenerate DOF is translational — the IMU absorbs it — but not when the degenerate DOF is rotational, where an unexplained confidence drop on an outer-loop run is still unresolved.

---

## The Gap

Without GPS, a robot must estimate its own pose entirely from onboard sensors — LiDAR, IMU, camera, wheel odometry — and small errors compound with every step. SLAM (Simultaneous Localisation and Mapping) addresses this by jointly estimating a trajectory and a map from sensor observations alone. The problem is that this only works as well as the sensors can constrain the estimate, and in several common environments — indoor buildings, tunnels, mines, urban canyons, subterranean and underwater spaces — they can't.

## What Makes an Environment Degenerate

A sensor is **degenerate** when it cannot constrain all 6 degrees of freedom of a pose:

| Environment | Sensor | Degenerate DOF |
|---|---|---|
| Long corridor | LiDAR | x (along the corridor) |
| Open field | LiDAR | x, y (both lateral directions) |
| Dark room | Camera | all |
| Flat ground | LiDAR | roll |

Unconstrained DOFs accumulate error at every step, with no natural mechanism to bound it.

This is diagnosable directly from the scan-matching optimization. At convergence, the cost function's Hessian **H** encodes how well-constrained each direction is: the estimate's covariance is Σ = H⁻¹, so a small eigenvalue of **H** means a large uncertainty in that direction. Three regimes make this concrete:

| Geometry | Eigenvalues (λ₁, λ₂) | Interpretation |
|---|---|---|
| Rich geometry | ≈ 850, ≈ 720 | Tight uncertainty ellipse — well constrained |
| Corridor | ≈ 4, ≈ 810 | Elongated ellipse along the corridor axis |
| Open plane | ≈ 3, ≈ 6 | Large ellipse in both directions — fully degenerate |

In a corridor, the walls constrain lateral position well, but the along-corridor direction has a near-zero Hessian eigenvalue: the cost function is flat along the direction of travel, and scan matching alone cannot tell you how far forward you've moved.

## Super Odometry: IMU as the Backbone

Super Odometry's response to this is architectural, not just a tuning choice: **IMU pre-integration is the primary propagation model**, and LiDAR and visual odometry supply corrections rather than the other way around. A degeneracy-detection stage scores each sensor's contribution using the eigenvalue test above, and that score adaptively scales the sensor's information matrix (Ω = Σ⁻¹) before it enters the factor-graph fusion step — degenerate directions are down-weighted rather than trusted equally with well-constrained ones.

The estimation pipeline alternates two update rates. IMU propagation runs at ~200 Hz:

```
x̂ₜ₊₁ = f(xₜ, u_IMU)
P̂ₜ₊₁ = F Pₜ Fᵀ + Q
```

LiDAR updates arrive at ~10 Hz via scan-to-map ICP/NDT:

```
K = P̂ Hᵀ (H P̂ Hᵀ + R_lidar)⁻¹
x⁺ = x̂ + K(z − h(x̂))
```

where R_lidar ∝ Σ = H_scan⁻¹ — inflated exactly when the scan is degenerate. This is also why the system still drifts in the worst case: if the smallest eigenvalue of H_scan is near zero, the Kalman gain K in that direction is near zero too, so the LiDAR update contributes nothing there and IMU bias integrates uncorrected.

## Experiments: Inner Loop vs. Outer Loop

Two indoor loop trajectories were run through Super Odometry, tracking a per-DOF confidence score from the degeneracy monitor:

| Run | Conf x | Conf y | Conf z | Roll | Pitch | Yaw | Result |
|---|---|---|---|---|---|---|---|
| Inner loop (sparse) | 0.43 | 1.00 | 1.00 | 0.99 | 1.00 | 0.86 | loop closes |
| Inner loop (dense) | 0.55 | 0.82 | 1.00 | 1.00 | 1.00 | 0.84 | loop closes |
| Outer loop (map capture) | 0.55 | 1.00 | 1.00 | **0.53** | 1.00 | 1.00 | anomalous |
| Outer loop (trajectory capture) | 0.72 | 1.00 | 1.00 | 0.94 | 1.00 | 1.00 | healthy |

<div align="center">
  <figure>
    <img src="/projects/assets/slam/super_odom_inside_loop_traj_2.png" alt="Inner loop trajectory, sparse room" width="440">
    <figcaption><em>Inner loop trajectory over a sparse point cloud. Confidence dips only on x (0.43) — the direction of travel — while z, roll, and pitch stay near 1.00.</em></figcaption>
  </figure>
  <figure>
    <img src="/projects/assets/slam/super_odom_inside_loop_traj_map_1.png" alt="Inner loop dense map" width="440">
    <figcaption><em>Same loop over the denser map. y-confidence drops to 0.82 here, likely a lateral section with fewer wall features, but the loop still closes cleanly.</em></figcaption>
  </figure>
</div>

**Inner loop succeeds** despite a low x-confidence (0.43–0.55, the corridor direction of travel) because the IMU carries translation in that direction while an equipment cluster (tables, chairs) anchors z, roll, and pitch strongly enough to keep the loop consistent — a small yaw drop (0.84–0.86) is the only cost. The resulting map captures the full room footprint with rich structural detail.

<div align="center">
  <figure>
    <img src="/projects/assets/slam/super_odom_outer_loop_map.png" alt="Outer loop map, roll confidence anomaly" width="440">
    <figcaption><em>Outer loop map capture — sparse, mostly doors and walls. Roll confidence drops to 0.53 here, the anomaly this experiment doesn't yet explain.</em></figcaption>
  </figure>
  <figure>
    <img src="/projects/assets/slam/super_odom_outer_loop_traj.png" alt="Outer loop trajectory" width="440">
    <figcaption><em>The trajectory capture of the same outer loop shows roll confidence at 0.94 — healthy, and inconsistent with the map capture's 0.53.</em></figcaption>
  </figure>
</div>

**The outer loop is where it breaks differently — and inconsistently.** The map-capture reading shows roll confidence dropping to 0.53, unexpected since roll isn't the degenerate DOF the corridor geometry predicts (that's normally x or y, not orientation). But the trajectory capture of what's labeled the same outer loop shows roll confidence at a healthy 0.94. Either these are two different passes through the loop rather than the same run viewed two ways, or the roll drop is localized to a specific segment the map capture caught and the trajectory capture didn't — the source data doesn't yet distinguish between these, so this is left as reported rather than reconciled. The outer-loop map itself shows comparatively poor detail regardless: a corridor with little geometry beyond doors and walls. The likely cause of the roll anomaly, where it appears, is floor irregularity or a tilted surface section, but this is not confirmed — it's an open question below, not a settled explanation.

## Key Insight: The Right DOFs, Not More Features

The refined hypothesis coming out of these runs: Super Odometry succeeds when the available geometry covers the *right* DOFs — not simply when there are more points in the cloud.

| Geometry present | DOFs constrained | IMU role |
|---|---|---|
| Vertical structure (walls, equipment) | z, roll, pitch | carries x, y |
| Floor plane only | z, pitch | carries x, y, roll |
| Corners, intersections | all 6 | backup only |
| Long corridor (parallel walls) | y, z, roll, pitch | must carry x |

The asymmetry that matters most: when the degenerate DOF is **translational**, the IMU absorbs it gracefully, since integrating acceleration to get position is exactly what it's built for. When the degenerate DOF is **rotational** — as the outer loop's roll anomaly may be — gyroscope bias integrates unchecked, and rotational drift compounds faster than translational drift. Unlike translational drift, it isn't recoverable by LiDAR alone without loop closure.

## Super Odometry vs. LIO-SAM: The Loop Closure Gap

| Capability | LIO-SAM | Super Odometry |
|---|---|---|
| Native loop closure | yes — pose graph + Scan Context | no |
| Global consistency after detection | yes | no — drift unbounded |
| Drift correction on revisit | yes | no |
| Degeneracy-aware weighting | no — uniform | yes |
| Adaptive sensor fusion | no | yes |
| Featureless robustness | limited | yes |

This is a structural difference, not a tuning gap: Super Odometry is a high-quality *odometry* system; LIO-SAM is a full *SLAM* system with loop closure built in. Comparing their loop-closure performance directly is really a question of system architecture, not which algorithm is better. The open research question is whether Super Odometry, paired with an external place-recognition module (Scan Context or DBoW), can match LIO-SAM's global consistency while keeping the degeneracy robustness LIO-SAM's uniform weighting doesn't have.

## Open Questions

1. **Loop closure parity.** Does Super Odometry + Scan Context match LIO-SAM on ATE once a loop is completed, and what shared metric should be used to compare them?
2. **Visual combination (VLIO).** In featureless environments, camera and LiDAR can degrade together. Does adaptive weighting give a meaningful edge here, or do both modalities fail simultaneously?
3. **The outer-loop roll anomaly.** Why does roll confidence drop to 0.53 on the outer loop but not the inner? Floor irregularity, tilted surfaces, and calibration drift are all still on the table.
4. **Uncertainty reporting.** Does Super Odometry's adaptive weighting genuinely reduce the reported covariance, or does it just suppress the sensor's contribution while the reported uncertainty stays falsely low? The latter would be dangerous downstream in a full system that trusts that covariance.
5. **Calibration.** Could extrinsic differences between this setup and the reference hardware Super Odometry was built on be confounding the comparison?

## Next Steps

Run LIO-SAM on the same rosbags as a control, agree on ATE on a known-loop sequence as the shared metric, and test VLIO against Super Odometry specifically in the featureless section where the roll anomaly showed up. This also means sharing calibration files (camera intrinsics, LiDAR-camera extrinsics, IMU noise parameters) so the comparison isn't confounded by setup differences, and aligning on benchmarking methodology before drawing conclusions either way.

The working hypothesis going in: LIO-SAM wins on loop closure by design, Super Odometry wins on raw odometry quality in degenerate sections, and the more interesting system combines both rather than picking one.
