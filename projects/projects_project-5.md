---
layout: page
title: "Planning with Scene Graphs on Spot"
---

# Planning with Scene Graphs on Spot

**Repo:** [karthiksomz/FARM-Project](https://github.com/karthiksomz/FARM-Project) \
**Stack:** ROS2 · Boston Dynamics Spot SDK · GraphNav · 3D Scene Graphs · SigLIP2 · Qwen3-VL · Viser \
**Status:** In progress

---
This project builds an end-to-end pipeline for language-driven navigation on Boston Dynamics' Spot: collect RGB-D trajectories through a real building, run them through the FARM scene-graph construction pipeline, and use the resulting 3D scene graph to drive GraphNav. A scene graph attaches semantics — captions, categories, spatial relations — on top of a metric map, so a natural-language query can resolve to a specific object instead of a hand-coded waypoint. The case that actually motivates the relational part of the graph is disambiguation: a room with several tables can't be resolved by category alone, but a query like *"go to the wooden table near the window, next to the blue cabinet"* chains spatial descriptors to pin down the one instance that's meant.

The pipeline currently works end-to-end on the simpler case — a visually unique object: querying *"a printer"* against a graph of ~1,900 objects returns the correct object in under 3 seconds, and Spot navigates to the closest feasible waypoint near it. The harder case above, disambiguating between several similar instances purely by spatial descriptors, is what the predicate system below is built for — and where it still falls short in practice (see `farm_cabinet`). A few problems remain open:

- **Closed-vocabulary detection.** YOLO backs the semantic layer, so it occasionally misclassifies objects outright — a 3D printer read as a refrigerator.
- **Noisy depth.** Camera depth isn't accurate enough, and despite care around occlusion and frame coverage, small spaces still introduce real positional uncertainty.
- **Fixed relation set.** The spatial predicates used to disambiguate objects (`On`, `Near`, `NextTo`, ...) are a hand-coded, closed set rather than learned or open-ended.
- **Heavy by design.** The retrieval ensemble leans on multiple VLMs and embedding models as fault tolerance against any one of them being wrong, which makes the pipeline expensive to run: north of 100GB VRAM unquantized, and 30 minutes to an hour to process a single rosbag over a 100m² space.

VLMs sit at the edges of this pipeline — parsing queries into predicates and verifying ambiguous matches — but do no explicit spatial reasoning themselves.

---

## Perception in Robotics

Classical robot navigation stacks operate on geometry: occupancy grids, point clouds, fixed waypoints. That's sufficient for "go to (x, y)" but not for "go to the printer" — there's no layer of the stack that knows what a printer is, or where one is, unless a human hard-codes it in advance. Closing that gap means giving the robot a semantic map that sits on top of the metric one: something that says not just *what's occupied* but *what's there*, so downstream planning and language grounding have something to query against.

## What Is a Scene Graph

A 3D scene graph represents a space as a graph of objects (nodes) with semantic attributes — category, caption, bounding box — connected by relations (edges) that capture things like containment, adjacency, or support ("the printer is on the desk," "the desk is in the office"). The nodes make the space *searchable* by description instead of by coordinate; the edges make it *reasoned about* — a planner can traverse "office → desk → printer" instead of treating every object as an unrelated point in space.

## A Short History of 3D Scene Graphs

Armeni et al. (2019) introduced the 3D scene graph as a semantic layer *on top of* the metric/topological maps SLAM already produces — nodes are objects with semantic attributes, edges are their spatial/temporal relations. That separation of "where things are" from "what's related to what" has held up in nearly every system since, this project included.

Two things have moved since: the **representation** — dense point clouds gave way to 3D Gaussian Splatting (Kerbl et al., 2023), adopted here at the object level (one Gaussian plus a sparse voxel cloud per object) rather than whole-scene — and the **vocabulary** — fixed taxonomies gave way to CLIP (Radford et al., 2021), letting nodes carry open-vocabulary captions/embeddings instead of a closed class label. One other distinction worth naming: offline mapping can afford heavier detectors and VLMs against a pre-recorded stream, while online mapping runs the same pipeline on-robot in real time, which is where cheaper detectors and async captioning start to matter.

### Relations After the Fact — and Where Pure Embeddings Fall Short

ConceptGraphs (Gu et al., 2024) is the clearest example of combining both threads with an LLM in the loop: build the open-vocabulary object set first, then prompt an LLM *after mapping* to infer the relational edges between objects from their captions and positions — no hand-coded relation logic, the LLM decides what's "on" what.

The gap that matters for querying, though, shows up even without any relations at all: **embedding-only retrieval can't handle relational composition.** Rank objects purely by similarity between the query text and each object's caption/visual embedding, and a query like *"the glass on the table next to the laptop"* collapses into a soup of similarity to "glass," "table," and "laptop" separately — nothing enforces that the top-ranked glass is actually **on** *that* table **and** that table is actually **next to** *that* laptop. If there are several glasses, on several tables, only one of which sits next to a laptop, embedding similarity alone has no mechanism to select it. This is precisely the failure mode behind `farm_cabinet` below and the disambiguation cases documented in [Replicating FARM in the Real World](/blog/post-3): the correct object is in the graph, correctly described, and still doesn't surface — because nothing checked whether the relation the query asked for actually holds in 3D.

## How FARM Handles Relations

FARM's answer is closer to the opposite of ConceptGraphs': it does **not** ask a VLM to invent relational edges between objects. Instead, the VLM/LLM's job is narrower and happens at *query time*, not at mapping time:

1. **Parse.** An LLM turns the free-text query into symbolic logic over a **fixed, hand-coded predicate set** — `Near`, `On`, `Above`, `Below`, `NextTo`, `Between`, `Inside`, `InRegion`, `LeftOf`, `RightOf`, `InFrontOf`, `Behind`, `Closest`, `Farthest`, `HasAttribute`, `IsCategory`.
2. **Evaluate geometrically.** Each predicate is checked directly against the 3D scene graph's object positions and extents — not guessed by a language model. `On(cabinet)` is a geometric test (vertical contact + horizontal overlap of two boxes), not an LLM's best guess from a caption.
3. **Verify on ambiguity.** Where geometry alone leaves close candidates, a VLM is used to adjudicate — the one place a model's judgment enters the relational decision at all.

| Predicate | Kind |
|---|---|
| `Near`, `NextTo`, `Between`, `Inside`, `InRegion` | proximity / containment |
| `On`, `Above`, `Below` | vertical / support |
| `LeftOf`, `RightOf`, `InFrontOf`, `Behind` | directional |
| `Closest`, `Farthest` | ranking over a candidate set |
| `HasAttribute`, `IsCategory` | attribute / category match (not relational — single-object) |

Everything the predicates take as arguments — the query target, the anchor objects, the attributes — stays fully open-vocabulary, resolved the same embedding/caption way as plain retrieval. The predicates are the only closed, fixed part of the system; what they're applied *to* is unrestricted free text. That split is the bet: exact, auditable geometric relations over an open-vocabulary object set, rather than trusting an LLM to both invent and apply the relation in one step.

## What Is the FARM Project

Captured trajectories (RGB, depth, camera info) are fed into the [FARM project's](https://github.com/karthiksomz/FARM-Project) pipeline, which builds the scene graph: it associates detections across frames into persistent 3D objects and generates captions and embeddings for each. Relations aren't stored as graph edges up front — per the predicate-based design above, they're evaluated at query time. In practice, on the current data, that query-time relational path isn't landing yet: the retrieval script's own log falls back explicitly when predicate execution fails —

<div class="callout callout-red" markdown="1">
```
Relational path unavailable ('NoneType' object has no attribute 'target_description');
falling back to embedding retrieval.
```
</div>

— meaning queries are currently resolved purely by embedding similarity between the query text and each object's caption/visual embedding, not by any graph structure connecting objects to each other. The graph exists as a flat set of richly-described nodes; the edges aren't reliably there yet.

## Pros and Cons

**What's working:** Retrieval over the flat object set is fast and reasonably precise even without relations. Querying a saved scene graph of 1,893 objects (`mistlab_complete_scene_1.pt`) for `"a printer"` returns 14 clusters in 2.68s, with the top candidates all correctly identifying printers by their generated captions:

| Rank | Score | Object ID | Caption |
|---|---|---|---|
| 1 | 0.160 | 2739 | "black and white boxy printer with visible cables" |
| 2 | 0.128 | 1342 | "black and white boxy printer with visible cables" |
| 3 | 0.111 | 1693 | "silver boxy printer with a logo and a white cable" |
| 4 | 0.098 | 182 | "black boxy printer with a green label" |
| 5 | 0.092 | 1170 | "gray boxy 3D printer with a hose attached" |

The retrieval pipeline is an ensemble rather than a single model — the log shows hits pooled from `caption_base`, `caption_task`, `siglip2`, `qwen3_vl`, and `qwen3_vl_task` (10 hits each), unioned into 34 unique candidates before clustering down to 14. Ensembling captioning and embedding models this way seems to be compensating for exactly the gap above: without relational structure to disambiguate, precision has to come from stacking multiple independent signals on the object's appearance alone.


## Demo Runs: Three Use Cases

Three demo runs on Spot test the core premise — retrieval alone suffices for a visually unique object, but recurring categories (chairs, tables, cabinets) need spatial predicates to disambiguate:

| Run | Query | Result |
|---|---|---|
| `farm_tripod` | `"tripod"` | <span class="status-badge status-success">Success</span> — unique object, retrieval alone suffices |
| `farm_rubber` | `"a stack of black rubber sheets near the small robots"` | <span class="status-badge status-partial">Partial Success</span> — resolved by the `Near(small robots)` predicate |
| `farm_cabinet` | `"file cabinet next to the tool workstation with a solar-system image on top of it"` | <span class="status-badge status-fail">Failure</span> — wrong cabinet; spatial grounding didn't hold between two ambiguous instances |

Full videos, field notes, and other failure modes (including outright YOLO misclassification, e.g. a 3D printer read as a refrigerator) are in [Replicating FARM in the Real World](/blog/post-3); the Spot setup used to run them is in [Recreating FARM on Spot: From Fiducial to Scene Graph](/blog/post-4).

## How It's Used in Planning

Resolving a query to an object's 3D bounding box, as above, is only half the loop — the other half is handing that box to something that can actually drive Spot. That something is GraphNav, Boston Dynamics' built-in topological planner: it represents the space as a graph of waypoints (states) linked by edges that store the relative transform between them, and plans by discrete search over those edges rather than continuous SLAM. Waypoints store locations, not the sensor data they were built from, which keeps the nav graph lightweight but also means it can't re-localize itself from nothing — restart the robot mid-mission and it loses track of where it sits in the graph unless it's re-anchored to something persistent in the environment, in practice a fixed fiducial marker.

Integrating the two graphs is mostly a frame-bookkeeping problem: an object's pose has to carry correctly from the scene graph's map frame into GraphNav's waypoint frame, and that chain of transforms has to stay consistent end to end or the robot drives to the wrong place with full confidence. The full walkthrough — fiducial setup, frame conventions, and where this breaks — is in [Recreating FARM on Spot: From Fiducial to Scene Graph](/blog/post-4).

Turning a retrieved object into a goal pose also needs more than its centroid — the centroid lies inside the object and can't serve as a goal. Instead, FARM searches the horizontal plane for a collision-aware standoff pose near the closest part of the target's voxel cloud: roughly 0.8 m away, never closer than 0.3 m, turned to face that point. A candidate is only accepted if it clears every other object's voxels by the robot radius plus a safety margin (0.3 m + 0.1 m by default); optional workspace bounds apply that same margin to walls, which the map doesn't store as objects, clamping the returned position inside them. If no clear pose exists within 2 m of the target, the system still returns its best attempt — flagged `navigable=False` — rather than silently handing off a goal it can't guarantee is safe.

<div align="center">
  <figure>
    <img src="/projects/assets/spot-scene-graph/viser_send_to_spot.png" alt="Viser viewer adapted to send a retrieved object's pose to Spot's planner" width="900">
    <figcaption>The Viser viewer adapted to close the loop on hardware: it takes the top-10 retrieval matches for a query, computes a navigable pose for the selected match, and sends it to Spot as a goal ("Sent #695 → x 5.27, y 2.15, yaw -138°") — falling back through nearby waypoints when a pose is flagged not-navigable, as shown here, rather than sending Spot somewhere it can't actually reach.</figcaption>
  </figure>
</div>

## Explore the Scene Graph

The screenshot above is from the Viser browser tool, not a purpose-built viewer. Since the scene state (`.pt`) and its background point cloud (`cloud.npz`) are just object boxes, captions, and points, they can be exported to a static file and rendered directly in the browser — no Viser session or vLLM backend required. Drag to orbit, scroll to zoom, and click a box to see its caption.

<!-- <div class="callout" markdown="1">
This is currently showing **placeholder data** — a synthetic room and generic boxes — while a real export is prepared. See <code>scripts/export_scene_graph_web.py</code> in the repo for how to point it at an actual <code>scene_state.pt</code>.
</div> -->

<div class="scene-viewer-embed">
  <iframe src="/projects/assets/spot-scene-graph/scene_viewer.html" loading="lazy" title="Interactive 3D scene graph viewer" allow="fullscreen"></iframe>
</div>

## Trying to Recreate It on Spot

With retrieval working, the open question is whether it can actually drive navigation on hardware, not just point at a bounding box in a viewer. The first trial run uses Boston Dynamics' own [GraphNav](https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/graph_nav_command_line) command-line example to build a nav graph and send Spot to a target pose. It works — but two things need care: getting the coordinate frame conventions right when converting a scene-graph object pose into a GraphNav waypoint, and closing loops through the anchors while recording, for GraphNav's anchoring optimization to reduce errors. Despite being an easy planner and control to use since it is inbuilt, the actions are discrete and the planner can only be used over trajectories that are visited, so if there is an object further from the scope of the trajectory despite the map showing that object.

## Open Questions

1. **Odometry on long trajectories.** Does switching to the Ouster's own IMU fix the drift seen on the bunker-trajectory bag, or is there a deeper extrinsic/sync issue?
2. **Predicate execution.** Why does query-time predicate evaluation fail on the current data, and is embedding-only fallback a fundamental ceiling, or just a stopgap until predicate execution is fixed?
3. **Pose error propagation.** How much of the scene graph's object-placement error is attributable to visual odometry specifically, versus the object detection/association stage?
4. **Query → waypoint.** What's the right way to convert a retrieved object's 3D bounding box into a GraphNav-compatible target pose, robust to the coordinate-frame and anchoring pitfalls already seen?

## What's Next

Re-collect the problem trajectory using the Ouster IMU and confirm odometry holds up over the full bunker loop. In parallel, dig into why FARM's query-time predicate execution isn't landing on this data, since that's the difference between "find object by appearance" and actually using the relation a query asked for. Once both are more stable, connect a real retrieval result end-to-end to a GraphNav waypoint and run a live "go to the printer" trial on the robot.

## References

1. Armeni, I., He, Z.-Y., Gwak, J., Zamir, A. R., Fischer, M., Malik, J., & Savarese, S. (2019). *3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera*. ICCV, pp. 5664–5673.
2. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. (CLIP)
3. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics (SIGGRAPH).
4. Gu, Q. et al. (2024). *ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning*. ICRA, pp. 5021–5028. [arXiv:2309.16650](https://arxiv.org/abs/2309.16650)
