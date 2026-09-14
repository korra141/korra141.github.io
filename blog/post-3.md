---
layout: post
title: "Replicating FARM in the Real World: What Works, What Doesn't"
date: 2026-09-04
---

We recently set out to reproduce **FARM (Find Anything using Relational
Spatial Memory)** end to end on a real Boston Dynamics Spot in our lab: build
a live 3D scene graph from Spot's RGB-D stream, query it in natural
language, and navigate to whatever the query resolves to. The pipeline
works — but getting it to work exposed exactly where a system like this
breaks in practice, which is arguably more useful than the successful runs.
This post is a short field report: three demo runs of different outcomes,
a look at the reconstructed scene graph itself, and a breakdown of the
failure modes we hit — perception, geometry, and, most consistently,
relational spatial grounding. (See [Planning with Scene Graphs on
Spot](/projects/projects_project-5) for the earlier project write-up this
follows on from.)

---

## The pipeline

1. **Segmentation + classification** — YOLOE (open-vocabulary) produces a
   mask and a category label for every detected object in each RGB-D frame.
2. **Embedding** — each mask is cropped and sent to DINOv3 to get an image
   embedding, used later for cross-frame correspondence and merging.
3. **Captioning** — the same crops are sent to a VLM (asynchronously) to
   get a structured caption: category, a short list of attributes, and a
   free-text description.
4. **Fusion** — detections are merged across frames into a single 3D
   Gaussian + sparse voxel cloud per object, using the embeddings and a
   union-find correspondence step, and linked into a scene graph via
   co-visibility.
5. **Query** — a natural-language query is parsed into a target category
   plus zero or more spatial predicates (`NextTo`, `HasAttribute`,
   `Near`, `Between`, …), matched against the graph, and returned as a
   ranked list of candidate objects with 3D coordinates.
6. **Navigation** — the coordinate of the selected match is handed to a
   navigation graph, which plans a collision-aware path and drives Spot to
   a standoff pose next to the target.

## The scene graph

<video controls muted playsinline poster="/blog/assets/farm/video/scene_graph_overview_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/scene_graph_overview_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/scene_graph_overview_1080p.mp4" type="video/mp4">
</video>

*The live scene graph, reconstructed from the lab's RGB-D stream. Every
translucent box is one tracked object, carrying a category, an attribute
list, a free-text caption, and a collision-aware navigation pose.*

This clip is a screen recording of our Viser-based scene-graph browser,
not a purpose-built visualization. Given the saved scene-state `.pt` (and
the `cloud.npz` background cloud), a sharper, annotated pass — or a small
interactive viewer embedded directly on the page — is a natural next step.

## Three demo runs

### 1. `farm_tripod` <span class="status-badge status-success">Success</span>

<video controls muted playsinline poster="/blog/assets/farm/video/farm_tripod_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/farm_tripod_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/farm_tripod_1080p.mp4" type="video/mp4">
</video>

Query: `"tripod"`. A single, visually distinctive object — retrieval
returns it at rank 1, the coordinate is read off the matched scene-graph
entry, and the navigation graph plans a direct route. Spot reaches it.

### 2. `farm_rubber` <span class="status-badge status-partial">Partial Success</span>

<video controls muted playsinline poster="/blog/assets/farm/video/farm_rubber_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/farm_rubber_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/farm_rubber_1080p.mp4" type="video/mp4">
</video>

Query: `"a stack of black rubber sheets near the small robots"`. Semantic
retrieval on `"black rubber sheets"` alone did not localize the stack —
it isn't a category YOLOE names cleanly, and its caption embedding wasn't
distinctive enough to rank correctly. Adding the relational clue
`Near(small robots)` narrowed the candidates enough to anchor a target
position about a metre from the true object — off, but close enough for
the navigation graph to plan a route that still lands somewhere usable.

### 3. `farm_cabinet` <span class="status-badge status-fail">Failure</span>

<video controls muted playsinline poster="/blog/assets/farm/video/farm_cabinet_poster.jpg" style="width:100%;height:auto;border-radius:8px;">
  <source src="/blog/assets/farm/video/farm_cabinet_1080p.webm" type="video/webm">
  <source src="/blog/assets/farm/video/farm_cabinet_1080p.mp4" type="video/mp4">
</video>

The lab has two file cabinets at opposite ends of the map. Query:
`"file cabinet next to the tool workstation with a solar-system image on
top of it"`, parsed into `NextTo(tool workstation)` and
`HasAttribute(solar-system image on top)`. Retrieval ranked the *wrong*
cabinet first and placed the correct one sixth — the rank-1 coordinate is
several metres from the intended object. The spatial predicates were
present in the parse but didn't move the correct instance up the ranking:
**retrieval score does not track spatial correctness**, and the system
could not use the relational description to disambiguate two instances of
the same category.

## What breaks, and where

### Perception: misclassification

YOLOE's open-vocabulary labels are frequently wrong in ways that look
arbitrary rather than "close but off": a 3D printer gets classified as a
**refrigerator**; a robotic arm gets classified as a **metallic pipe**.
Neither is a near-miss on the label taxonomy — both suggest the visual
feature the classifier is keying on (a boxy metal enclosure, a jointed
metal profile) generalizes past the category boundary we actually care
about. Because the downstream caption and attribute list are built off
the same crop, a misclassified object often carries a plausible-sounding
but wrong caption all the way into the scene graph.

We also see the same failure show up as a *confidence* problem rather
than an outright wrong label. QR/AprilTag fiducial markers used for
localization get detected with wildly inconsistent uncertainty — some
low, some high, for what is visually the same flat printed marker.
Door handles show a clearer pattern: the ones physically close to a
fiducial marker are captured with low uncertainty, while the same object
category further from any marker is captured with high uncertainty. That
tracks with the geometry problem below — objects far from a localization
anchor inherit a noisier pose estimate, and that noise appears to leak
into how confidently the object is captioned and merged.

### Geometry: depth accuracy

Depth from the RGB-D stream is not uniformly trustworthy. In the frames
we inspected, object positions can be off by as much as 3–5 m in the
worst cases — enough to place a box in a different part of the room than
the physical object. This is the direct cause of `farm_cabinet`'s rank-1
result landing on open floor rather than on a cabinet, and it compounds
with the fiducial-distance effect above: the further a frame is from a
localization anchor, the less its depth (and therefore its 3D position)
can be trusted.

### Relational spatial grounding — the main bottleneck

<div class="callout callout-red" markdown="1">
This is the result we think matters most for the FARM thesis. The paper's
premise is that spatial connectors — `near`, `next to`, `between` — let
the system disambiguate frequently-recurring categories like chairs and
tables that plain semantic retrieval can't tell apart. In practice, that
disambiguation step is where the pipeline struggles hardest, even with
redundant multi-modal evidence (DINO embedding + caption + attributes)
backing every object.
</div>

Beyond the recorded `farm_cabinet` case, we saw the same pattern
qualitatively on other queries during testing:

- **"the table with the tools on it"** — the correct table is not
  retrieved at all; the spatial predicate doesn't surface it over
  visually-similar tables that don't satisfy the relation.
- **"the grey chair between the two dark blue cabinets"** — a three-way
  relation (`Between`) rather than a pairwise one. None of the top
  retrieved candidates matched the object the query was actually
  describing; the correct chair didn't surface even loosely.
- **"the solar-system painting with the black frame on the cabinet"** —
  querying for the *painting* itself, via its own attributes
  (solar-system imagery, black frame) plus `On(cabinet)`, fails outright:
  the object isn't found at all. This is the same relation from the
  `farm_cabinet` case (the poster is what distinguishes the correct
  cabinet, ranked 6th) approached from the other direction — asking for
  the small attribute-object using its relation to the larger one it
  sits on, rather than the other way around — and it fails the same way.
- **"the cabinet with the painting [on it / near it]"** — instead of
  finding the cabinet actually positioned near a painting, the system
  returns whichever cabinet instance has the lowest retrieval
  uncertainty overall, regardless of whether the spatial relation holds.

The pattern across all four: `near`, `on`, and `between` are parsed
correctly into predicates, but the predicates don't reliably re-rank
candidates by whether the spatial relation actually holds in 3D. The
system falls back to whichever candidate has the strongest semantic match
on its own, independent of the relation the query asked for. That's
consistent with `farm_cabinet`: the correct cabinet exists in the graph,
with the right attributes, at the right location — it's ranked 6th
anyway.

## Reproducing this

The code is at
[karthiksomz/FARM-Project](https://github.com/karthiksomz/FARM-Project).
The offline driver (`scene_graph.offline.run`) takes a FARM-Scenes,
ScanNet, or `.sens` capture and produces a saved scene state (`.pt`);
`scripts/view_scene_state.py --pt <file>` opens it in the same
Viser-based browser used for the recording above, including the query
panel. See `EVALUATION.md` in the repo for the full benchmark protocol if
you want numbers rather than anecdotes.
