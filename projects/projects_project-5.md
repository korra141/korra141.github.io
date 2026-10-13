---
layout: page
title: "Planning with Scene Graphs on Spot"
---

# Planning with Scene Graphs on Spot

**Repo:** [karthiksomz/FARM-Project](https://github.com/karthiksomz/FARM-Project) \
**Stack:** ROS2 · Boston Dynamics Spot SDK · GraphNav · 3D Scene Graphs · SigLIP2 · Qwen3-VL · Viser \
**Status:** In progress

---
This project builds an end-to-end pipeline for language-driven navigation on Boston Dynamics' Spot: It uses the [FARM-Project](https://github.com/GoldenGait/FARM-Project) to build the scene graph from RGB-D images combining it with Spot's NavGraph to reach standoff pose. A scene graph attaches semantics — captions, categories, spatial relations — on top of a metric map, so a natural-language query can resolve to a specific object instead of a hand-coded waypoint. The case that actually motivates FARM's scene graph is disambiguation: a room with several tables can't be resolved by category alone, but a query like *"go to the wooden table near the window, next to the blue cabinet"* chains spatial descriptors to pin down the one instance that's meant.

This project analyses the FARM Project in real indoor setting by deploying its offline-mapping configuration, on a Boston Dynamics Spot quadruped. It examines the engineering that published systems rarely report: the work needed to connect a relational object-retrieval memory to a downstream planner so that the robot can actually navigate to the objects it retrieves.

## Perception in Robotics

Classical robot navigation stacks operate on geometry: occupancy grids, point clouds, fixed waypoints. That's sufficient for "go to (x, y)" but not for "go to the printer" — there's no layer of the stack that knows what a printer is, or where one is, unless a human hard-codes it in advance. Closing that gap means giving the robot a semantic map that sits on top of the metric one: something that says not just *what's occupied* but *what's there*, so downstream planning and language grounding have something to query against.

## What Is a Scene Graph

A 3D scene graph represents a space as a graph of objects (nodes) with semantic attributes — category, caption, bounding box — connected by relations (edges) that capture things like containment, adjacency, or support ("the printer is on the desk," "the desk is in the office"). The nodes make the space *searchable* by description instead of by coordinate; the edges make it *reasoned about* — a planner can traverse "office → desk → printer" instead of treating every object as an unrelated point in space.

## A Short History of 3D Scene Graphs

Armeni et al. (2019) introduced the 3D scene graph as a semantic layer *on top of* the metric/topological maps SLAM already produces — nodes are objects with semantic attributes, edges are their spatial/temporal relations. That separation of "where things are" from "what's related to what" has held up in nearly every system since, this project included.

Two things have moved since: the **representation** — dense point clouds gave way to 3D Gaussian Splatting (Kerbl et al., 2023), adopted here at the object level (one Gaussian plus a sparse voxel cloud per object) rather than whole-scene — and the **vocabulary** — fixed taxonomies gave way to CLIP (Radford et al., 2021), letting nodes carry open-vocabulary captions/embeddings instead of a closed class label. One other distinction worth naming: offline mapping can afford heavier detectors and VLMs against a pre-recorded stream, while online mapping runs the same pipeline on-robot in real time, which is where cheaper detectors and async captioning start to matter.

## What Is the FARM Project

Captured trajectories (RGB, depth, camera info) are fed into the [FARM project's](https://github.com/karthiksomz/FARM-Project) pipeline, which builds the scene graph. It helps to start with what actually ends up stored in the map before looking at how a query resolves against it.

**What's in a node.** Every detected object becomes a node carrying a category (from YOLO's open-vocabulary detector), a generated caption and attribute list, and a geometric representation — a 3D Gaussian plus a sparse voxel cloud per object, fused across frames and linked into the graph via co-visibility. Relations between objects are not stored up front; they're computed later, at query time, against this flat object set.

**Query parsing.** A free-text query is first parsed into symbolic logic: a target, zero or more anchor objects, and spatial relation predicates drawn from a fixed, closed set of 16 — `Near, On, Above, Below, NextTo, Between, Inside, InRegion, LeftOf, RightOf, InFrontOf, Behind, Closest, Farthest, HasAttribute, IsCategory`. This parsing is done by Qwen3.5-9B, the same model that generates captions during mapping. A separate model, Qwen3-Embedding, embeds the query text for retrieval. (`LeftOf`/`RightOf` are the predicates worth pausing on — they implicitly assume a viewpoint, and whose viewpoint that is isn't obvious from the query alone.)

**Retrieval.** Targets and anchors are matched against the graph by an ensemble rather than a single model: hits are pooled from `caption_base`, `caption_task`, `siglip2`, `qwen3_vl`, and `qwen3_vl_task` embeddings, unioned into a candidate pool, then clustered down to a ranked shortlist. Querying a saved scene graph of 1,893 objects for `"a printer"`, for example, returns 14 clusters from a union of 34 hits in under three seconds, with the top five all correctly identifying printers by their generated captions.

**Why the redundancy.** The system currently leans on more models than strictly necessary, and that's deliberate rather than an oversight: YOLO and the captioner are two independent votes on an object's category, which helps when YOLO's open-vocabulary label is wrong — the cost is that they can disagree, and in the default tiered mode a confident-but-wrong YOLO class still wins over a correct caption. The same logic extends to stacking multiple embedding models in the retrieval ensemble above: without relational structure to lean on, precision has to come from several independent signals on an object's appearance agreeing, rather than from one model being right. The result is a pipeline that is currently disjoint — several models doing overlapping jobs with no single piece owning "is this the right object" — which is the direct cost side of the fault-tolerance bet (see *Compute & Scaling* below).

**The VLM as checker, not reasoner.** The most important design decision in FARM is how little a model is trusted to decide on its own. Unlike ConceptGraphs (Gu et al., 2024), which builds the open-vocabulary object set first and then prompts an LLM *after mapping* to infer relational edges between objects from their captions and positions, FARM never asks a model to invent relations. Instead, a model's judgment enters at query time only, and only in a narrow, scoped way:

1. **Parse.** An LLM turns the free-text query into symbolic logic over the fixed predicate set above — this is the query-parsing step already described.
2. **Evaluate geometrically.** Each predicate is checked directly against the 3D scene graph's object positions and extents, not guessed by a language model. `On(cabinet)` is a geometric test (vertical contact + horizontal overlap of two boxes), not an LLM's best guess from a caption.
3. **Verify on ambiguity.** Where geometry alone leaves close candidates, a VLM is used to adjudicate — the one place a model's judgment enters the relational decision at all.

| Predicate | Kind |
|---|---|
| `Near`, `NextTo`, `Between`, `Inside`, `InRegion` | proximity / containment |
| `On`, `Above`, `Below` | vertical / support |
| `LeftOf`, `RightOf`, `InFrontOf`, `Behind` | directional |
| `Closest`, `Farthest` | ranking over a candidate set |
| `HasAttribute`, `IsCategory` | attribute / category match (not relational — single-object) |

Everything the predicates take as arguments — the query target, the anchor objects, the attributes — stays fully open-vocabulary, resolved the same embedding/caption way as plain retrieval. The predicates are the only closed, fixed part of the system; what they're applied *to* is unrestricted free text. That split is the bet: exact, auditable geometric relations over an open-vocabulary object set, with a VLM checking edge cases rather than deciding relations outright.

<div class="callout" markdown="1">
**Why checker beats reasoner.** FARM's own ablation compares a VLM used as an active relational reasoner — deciding relations directly, the way ConceptGraphs' post-mapping LLM does — against a VLM used only as a checker on geometrically-shortlisted candidates. <!-- TODO: insert VLM-checker vs. VLM-relational-reasoner ablation numbers and the specific failure mode this fixes, once available --> *Numbers to follow.*
</div>

## Issues with Scene Graphs: Temporal Consistency

The scene graph FARM builds is a snapshot of one captured trajectory, not a live, continuously-updated model of the space. Mapping runs offline per rosbag — there is no mechanism in the current pipeline for merging a new capture into an existing scene graph, reconciling an object that's moved since the last mapping run, or marking a node stale once the space it describes has changed. A node's caption, embedding, and geometry are only ever as current as the capture that produced them.

<!-- TODO: confirm against the current implementation — does a re-run always build a fresh scene graph from scratch, or is there any cross-session merge/versioning logic at all? What actually happens to a node when the object it describes is physically moved or removed between runs? -->

## Compute & Scaling (Why FARM Is Heavy)

The redundancy described above is not free. The retrieval ensemble leans on multiple VLMs and embedding models simultaneously as fault tolerance against any one of them being wrong, which makes the pipeline expensive to run: north of 100GB VRAM unquantized, and 30 minutes to an hour to process a single rosbag over a 100m² space. That cost is the direct trade for the fault-tolerance bet described in *What Is the FARM Project* above — more independent signals means more models running per frame, not a smarter single model.

This is also why FARM currently only makes sense offline. The offline/online distinction drawn earlier is where this cost actually bites: offline mapping can afford to run heavy detectors and VLMs against a pre-recorded stream with no latency constraint, while an online version of the same pipeline running on-robot in real time would need cheaper detectors and asynchronous captioning to stay usable — neither of which FARM does today.

## Demo Runs: Three Use Cases

Three demo runs on Spot test the core premise — retrieval alone suffices for a visually unique object, but recurring categories (chairs, tables, cabinets) need spatial predicates to disambiguate:

| Run | Query | Result |
|---|---|---|
| `farm_tripod` | `"tripod"` | <span class="status-badge status-success">Success</span> — unique object, retrieval alone suffices |
| `farm_rubber` | `"a stack of black rubber sheets near the small robots"` | <span class="status-badge status-partial">Partial Success</span> — resolved by the `Near(small robots)` predicate |
| `farm_cabinet` | `"file cabinet next to the tool workstation with a solar-system image on top of it"` | <span class="status-badge status-fail">Failure</span> — wrong cabinet; spatial grounding didn't hold between two ambiguous instances |

Full videos, field notes, and other failure modes (including outright YOLO misclassification, e.g. a 3D printer read as a refrigerator) are in [Replicating FARM in the Real World](/blog/post-3); the Spot setup used to run them is in [Recreating FARM on Spot: From Fiducial to Scene Graph](/blog/post-4). That post also carries the fuller breakdown of what works and what doesn't — including how this compares to ConceptGraphs' post-hoc relational inference and why embedding-only retrieval can't compose relations on its own.

## Results: Planning Over a Discrete Nav Graph

Resolving a query to an object's 3D bounding box, as above, is only half the loop — the other half is handing that box to something that can actually drive Spot, and reading the three demo runs above requires understanding the constraints of that second half as much as the first.

That something is GraphNav, Boston Dynamics' built-in topological planner: it represents the space as a graph of waypoints (states) linked by edges that store the relative transform between them, and plans by discrete search over those edges rather than continuous SLAM. Waypoints store locations, not the sensor data they were built from, which keeps the nav graph lightweight but also means it can't re-localize itself from nothing — restart the robot mid-mission and it loses track of where it sits in the graph unless it's re-anchored to something persistent in the environment, in practice a fixed fiducial marker. Without that fiducial, anchoring the nav graph to the scene graph's map frame can misbehave in ways that look like a planning bug but are actually a localization one.

Discrete search over a fixed waypoint graph also means the plan can only go where the graph already has points: if the scene graph resolves a query to a location the nav graph never captured a waypoint near, there is no path to it regardless of how correct the retrieval was. The two graphs only agree where their coverage overlaps.

Integrating the two graphs is mostly a frame-bookkeeping problem: an object's pose has to carry correctly from the scene graph's map frame into GraphNav's waypoint frame, and that chain of transforms has to stay consistent end to end or the robot drives to the wrong place with full confidence. The full walkthrough — fiducial setup, frame conventions, and where this breaks — is in [Recreating FARM on Spot: From Fiducial to Scene Graph](/blog/post-4).

Turning a retrieved object into a goal pose also needs more than its centroid — the centroid lies inside the object and can't serve as a goal. Instead, FARM searches the horizontal plane for a collision-aware standoff pose near the closest part of the target's voxel cloud: roughly 0.8 m away, never closer than 0.3 m, turned to face that point. A candidate is only accepted if it clears every other object's voxels by the robot radius plus a safety margin (0.3 m + 0.1 m by default); optional workspace bounds apply that same margin to walls, which the map doesn't store as objects — there is no way to mark points *inside* a wall as non-navigable beyond that outer workspace boundary. If no clear pose exists within 2 m of the target, the system still returns its best attempt — flagged `navigable=False` — rather than silently handing off a goal it can't guarantee is safe.

<div align="center">
  <figure>
    <img src="/projects/assets/spot-scene-graph/viser_send_to_spot.png" alt="Viser viewer adapted to send a retrieved object's pose to Spot's planner" width="900">
    <figcaption>The Viser viewer adapted to close the loop on hardware: it takes the top-10 retrieval matches for a query, computes a navigable pose for the selected match, and sends it to Spot as a goal ("Sent #695 → x 5.27, y 2.15, yaw -138°") — falling back through nearby waypoints when a pose is flagged not-navigable, as shown here, rather than sending Spot somewhere it can't actually reach.</figcaption>
  </figure>
</div>

The screenshot above is from the Viser browser tool, not a purpose-built viewer. Since the scene state (`.pt`) and its background point cloud (`cloud.npz`) are just object boxes, captions, and points, they can be exported to a static file and rendered directly in the browser — no Viser session or vLLM backend required. Drag to orbit, scroll to zoom, and click a box to see its caption.

<!-- <div class="callout" markdown="1">
This is currently showing **placeholder data** — a synthetic room and generic boxes — while a real export is prepared. See <code>scripts/export_scene_graph_web.py</code> in the repo for how to point it at an actual <code>scene_state.pt</code>.
</div> -->

<div class="scene-viewer-embed">
  <iframe src="/projects/assets/spot-scene-graph/scene_viewer.html" loading="lazy" title="Interactive 3D scene graph viewer" allow="fullscreen"></iframe>
</div>

## Small Improvements

1. **Odometry on long trajectories.** A lot of the error compounds in the localisation being estimated using visual inertial, adding lidar can improve the graph localisation of objects.

2. Reducing the redundancies in the overlapping models described in *Compute & Scaling* above.

3. **Closed semantics with YOLO.** The pipeline relies on YOLO to categorise the objects since yolo is a closed object detector, things like 3d printer get seen as refrigerator and vicon systems are fire alarms which are terrible. 
4. **Query → waypoint.** What's the right way to convert a retrieved object's 3D bounding box into a GraphNav-compatible target pose, robust to the coordinate-frame and anchoring pitfalls already seen?

## Big Improvements

2. **Predicate execution.** Given it is a closed set of spatial predicates which are applied based on geometry, they fail for complex tasks as shown in the demo, moving from handcoded list to a more intelligent solution.

## What's Next

The deeper question — why FARM's VLM-as-checker design works where it works, and why it still fails on cases like `farm_cabinet` — along with the VLM-as-checker ablation results and the missing piece that points to, is better answered with evidence than speculation. That analysis, plus the follow-up experiments it needs, belongs in [Replicating FARM in the Real World](/blog/post-3) as those experiments land, rather than duplicated here.

## References

1. Armeni, I., He, Z.-Y., Gwak, J., Zamir, A. R., Fischer, M., Malik, J., & Savarese, S. (2019). *3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera*. ICCV, pp. 5664–5673.
2. Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. (CLIP)
3. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics (SIGGRAPH).
4. Gu, Q. et al. (2024). *ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning*. ICRA, pp. 5021–5028. [arXiv:2309.16650](https://arxiv.org/abs/2309.16650)
