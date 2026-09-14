---
layout: page
title: "Planning with Scene Graphs on Spot"
---

# Planning with Scene Graphs on Spot

**Repo:** [karthiksomz/FARM-Project](https://github.com/karthiksomz/FARM-Project) \
**Stack:** ROS2 · Boston Dynamics Spot SDK · GraphNav · 3D Scene Graphs · SigLIP2 · Qwen3-VL · Viser \
**Status:** In progress

---

> Robots that navigate by fixed waypoints can't be asked to "go to the printer" — they have no representation of what's in the space beyond geometry. A 3D scene graph closes that gap by attaching semantics (captions, categories, relations) to a metric map, so a natural-language query can resolve to a real object and, eventually, a real waypoint. This project builds that pipeline on Spot: collecting RGB-D trajectories through a real building, feeding them through the FARM scene-graph construction pipeline, and testing whether the result can drive Boston Dynamics' GraphNav. It currently works end-to-end for retrieval — querying "a printer" against a graph of ~1,900 objects returns the right object in under 3 seconds — but two problems are still open: odometry drift on longer trajectories corrupts the map, and the scene graph isn't yet recovering *relations* between objects, only their individual embeddings.

---

## Perception in Robotics

Classical robot navigation stacks operate on geometry: occupancy grids, point clouds, fixed waypoints. That's sufficient for "go to (x, y)" but not for "go to the printer" — there's no layer of the stack that knows what a printer is, or where one is, unless a human hard-codes it in advance. Closing that gap means giving the robot a semantic map that sits on top of the metric one: something that says not just *what's occupied* but *what's there*, so downstream planning and language grounding have something to query against.

## What Is a Scene Graph

A 3D scene graph represents a space as a graph of objects (nodes) with semantic attributes — category, caption, bounding box — connected by relations (edges) that capture things like containment, adjacency, or support ("the printer is on the desk," "the desk is in the office"). The nodes make the space *searchable* by description instead of by coordinate; the edges make it *reasoned about* — a planner can traverse "office → desk → printer" instead of treating every object as an unrelated point in space.

## What Is the FARM Project

Captured trajectories (RGB, depth, camera info) are fed into the [FARM project's](https://github.com/karthiksomz/FARM-Project) pipeline, which builds the scene graph: it associates detections across frames into persistent 3D objects, generates captions and embeddings for each, and — in principle — infers the relational edges between them. In practice, on the current data, that last part isn't landing yet: the retrieval script's own log falls back explicitly when it can't find a relational path —

```
Relational path unavailable ('NoneType' object has no attribute 'target_description');
falling back to embedding retrieval.
```

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

**What's not working:**
- **Pose error from visual odometry** propagates directly into the scene graph — if the trajectory the objects were localized against is wrong, the objects are wrong too, independent of how good the captioning/embedding side is.
- **No spatial object connections.** As above — the relational path is unavailable, so the graph can't yet answer anything that depends on how objects relate to each other, only "which object looks most like this query."
- **Odometry/mapping failure on the longer trajectory.** One collected bag — through a bunker-like building — produced incorrect odometry and mapping. The likely cause is not using the Ouster's onboard IMU (a VectorNav IMU was used instead on that run); the fix going forward is to make sure the Ouster IMU is what's driving odometry, not a secondary IMU that isn't as tightly coupled to the LiDAR.

*(Nuance from the demo runs below: the relational path isn't entirely dead — spatial predicates do get parsed and do get matched against candidates. What's missing is that they don't reliably re-rank candidates by whether the spatial relation actually holds in 3D, which is functionally close to "unavailable" but not quite the same failure.)*

## Demo Runs: Three Use Cases

The core premise FARM is testing is that plain semantic retrieval is enough when an object is visually **unique** in the scene, but breaks down on categories that recur — a chair, a table, a cabinet — where many instances share a caption and only their spatial relationships to other objects can tell them apart. Three demo runs on Spot span that spectrum, from a fully unique object to a category with real duplicates in the space.

Before that split, though, it's worth being clear about a separate failure mode that affects *any* object, unique or not: **outright misclassification from YOLO's open-vocabulary labelling**. This isn't a near-miss on the category — the detector keys on a coarse visual feature (a boxy enclosure, a metallic jointed profile) that generalizes past the object it's actually looking at. In our testing this produced labels like a Vicon motion-capture camera called a **fire alarm**, a robotic arm called a **metal pipe**, and a 3D printer called a **refrigerator**. Since the caption and attribute list are generated off the same crop as the (wrong) label, a misclassified object carries a plausible but incorrect description all the way into the scene graph — no amount of relational reasoning fixes a query if the node itself is mislabeled at the source.

### 1. `farm_tripod` — unique object, retrieval alone succeeds

<video controls muted playsinline poster="/blog/assets/farm/video/farm_tripod_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/farm_tripod_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/farm_tripod_1080p.mp4" type="video/mp4">
</video>

Query: `"tripod"`. There's exactly one tripod in the space, visually distinct from everything around it — no spatial disambiguation is needed. Semantic retrieval alone returns it at rank 1, and Spot navigates straight to it. This is the ceiling case: when an object is unique enough that its caption/embedding alone separates it from every other node, the pipeline works about as well as it can.

### 2. `farm_rubber` — an ambiguous object resolved by a spatial predicate

<video controls muted playsinline poster="/blog/assets/farm/video/farm_rubber_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/farm_rubber_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/farm_rubber_1080p.mp4" type="video/mp4">
</video>

Query: `"a stack of black rubber sheets near the small robots"`. Semantic retrieval on "black rubber sheets" alone could not localize the stack directly — it isn't a category YOLO labels cleanly, and the caption embedding wasn't distinctive enough on its own to rank it correctly. This is where the relational side of FARM is supposed to earn its keep: adding the spatial predicate `Near(small robots)` gave the system a second, independent signal to anchor on, and it successfully identified the correct stack — a case where spatial grounding did what the paper claims it should.

### 3. `farm_cabinet` — two ambiguous instances, spatial grounding fails

<video controls muted playsinline poster="/blog/assets/farm/video/farm_cabinet_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/farm_cabinet_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/farm_cabinet_1080p.mp4" type="video/mp4">
</video>

The lab has two file cabinets at opposite ends of the map — a real case of a recurring category, exactly the situation spatial predicates are meant to resolve. Query: `"file cabinet next to the tool workstation with a solar-system image on top of it"`, parsed into `NextTo(tool workstation)` and `HasAttribute(solar-system image on top)`. The system goes to the *wrong* cabinet initially. The likely cause is twofold: either the correct cabinet's pose/caption carries high enough uncertainty that it gets outranked outright, or the pipeline never associates the solar-system poster sitting on top of the cabinet as a relation *to* the cabinet in the first place — if that `On(cabinet, poster)` edge is never formed, there's nothing for the query's `HasAttribute` clause to match against, and retrieval falls back to whichever cabinet instance looks most generically like "a cabinet."

Together, these three runs are the clearest evidence for where the FARM thesis holds and where it doesn't yet: uniqueness alone is enough for `farm_tripod`; a single spatial predicate rescued `farm_rubber`; but `farm_cabinet` shows spatial grounding failing exactly where it matters most — two genuinely ambiguous instances of the same category, disambiguated only by a relation the pipeline couldn't reliably form or weigh. See [Replicating FARM in the Real World](/blog/post-3) for the full field report, including the scene-graph reconstruction and additional queries that hit the same relational bottleneck.

## How It's Used in Planning

The scene graph is meant to be the bridge between a language query and a physical target: resolve "a printer" to object 2739's 3D bounding box, then hand that box's centroid to a motion planner as a goal pose. A [Viser](https://github.com/nerfstudio-project/viser)-based viewer makes this loop inspectable — the scene renders as translucent, color-coded bounding boxes over the mapped trajectory, clicking an object surfaces its caption JSON (category, supercategory, attributes, description, decision) and a cropped image, and a query panel lets you type a free-text search and — once a vLLM backend is started — run retrieval live in the browser rather than from the command line. A max-box-side filter is there mainly to prune degenerate boxes (huge false-positive volumes) out of the view. This tool is currently more useful for debugging the pipeline than for planning itself, but it's the same retrieval path a planner would call.

<div align="center">
  <figure>
    <img src="/projects/assets/spot-scene-graph/viser_scene_viewer.png" alt="Viser scene graph viewer showing bounding boxes over a mapped trajectory" width="900">
    <figcaption>The Viser viewer over a mapped corridor trajectory. Each translucent box is a detected object; the red/green line is the recorded trajectory through the space. The right panel shows a clicked object's caption JSON, its cropped image, and the query panel for running retrieval directly in the browser.</figcaption>
  </figure>
  <figure>
    <img src="/projects/assets/spot-scene-graph/viser_query_panel.png" alt="Viser query panel showing a live 'a printer' search and object 2739's populated caption JSON" width="320">
    <figcaption>The query panel after searching "a printer" live in the browser — the top-5 ranking matches the CLI script exactly (see table above). Object 2739's caption JSON is now fully populated: category "printer", supercategory "appliance", attributes ["black", "white", "boxy", "with cables"], and decision "keep" — this is what a resolved node looks like once FARM's captioning stage has actually run on it, versus the empty template seen before captioning.</figcaption>
  </figure>
</div>

## Trying to Recreate It on Spot

With retrieval working, the open question is whether it can actually drive navigation on hardware, not just point at a bounding box in a viewer. The first trial run uses Boston Dynamics' own [GraphNav](https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples/graph_nav_command_line) command-line example to build a nav graph and send Spot to a target pose. It works — but two things need care: getting the coordinate frame conventions right when converting a scene-graph object pose into a GraphNav waypoint, and not closing loops through the anchors while recording, since that appears to interfere with how GraphNav's anchoring optimization resolves the graph.

## Open Questions

1. **Odometry on long trajectories.** Does switching to the Ouster's own IMU fix the drift seen on the bunker-trajectory bag, or is there a deeper extrinsic/sync issue?
2. **Relational edges.** Why is the relational path unavailable in FARM's output, and is embedding-only retrieval a fundamental ceiling, or just a stopgap until relations are fixed?
3. **Pose error propagation.** How much of the scene graph's object-placement error is attributable to visual odometry specifically, versus the object detection/association stage?
4. **Query → waypoint.** What's the right way to convert a retrieved object's 3D bounding box into a GraphNav-compatible target pose, robust to the coordinate-frame and anchoring pitfalls already seen?

## What's Next

Re-collect the problem trajectory using the Ouster IMU and confirm odometry holds up over the full bunker loop. In parallel, dig into why FARM isn't producing relational edges, since that's the difference between "find object by appearance" and an actual scene *graph*. Once both are more stable, connect a real retrieval result end-to-end to a GraphNav waypoint and run a live "go to the printer" trial on the robot.
