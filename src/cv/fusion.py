"""
SmartForge — Multi-Image Fusion (Batch 2)
==========================================
Fuses parallel per-image CV detections into a deduplicated damage list
using a NetworkX directed graph as the in-memory "Graph Database".

Graph schema
------------
Nodes
    IMAGE      — one per source image (keyed "IMG_<idx>")
    PART       — one per unique car part (keyed "PART_<name>")
    DETECTION  — one per raw bounding box (keyed by detection_id)

Edges
    IMAGE  → DETECTION   (relation: "contains")
    DETECTION → PART     (relation: "located_on")

De-duplication rule
-------------------
All DETECTION nodes sharing the same PART node represent the SAME physical
damage seen from different camera angles.  The detection with the highest
YOLO confidence becomes the "Golden Record".  Additional metadata:
    visibility_count   — how many images confirmed this damage
    seen_in_indices    — exact image indices for the audit trail
    primary_image_idx  — which image gives the best (highest-conf) view

Image-recycling fraud detection
--------------------------------
If the same damage fingerprint (identical bounding box coordinates) appears
on DETECTION nodes linked to DIFFERENT IMAGE nodes, the claimant submitted
the same photo multiple times.  This flags a fraud_recycling_flag on the
Golden Record and adds a message to fraud_report["flags"].

Global state
------------
    claims_graph : nx.DiGraph
        Populated by build_claims_graph() and fuse_detections().
        Queryable after any pipeline run for audit purposes:

            # all detections on Front Bumper
            list(claims_graph.predecessors("PART_Front Bumper"))

            # source image for detection W0-001
            list(claims_graph.predecessors("W0-001"))

            # check for recycling flags
            [n for n, d in claims_graph.nodes(data=True)
             if d.get("fraud_recycling_flag")]

Public API
----------
    build_claims_graph(all_detections) → nx.DiGraph
    fuse_detections(all_detections)    → (fused_list, recycling_flags, graph_stats)
"""

from typing import Any, Dict, List, Tuple

import networkx as nx

# ── Module-level global — reset per pipeline job by build_claims_graph() ─────
claims_graph: nx.DiGraph = nx.DiGraph()


def build_claims_graph(
    all_detections: List[Dict[str, Any]],
) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from a flat list of detection dicts.

    Each detection dict must contain at minimum:
        detection_id        str
        source_image_index  int
        source_image_path   str
        location            str  (car part name)

    Returns the populated graph and also updates the module-level
    `claims_graph` singleton.
    """
    global claims_graph

    G = nx.DiGraph()
    claims_graph = G

    for det in all_detections:
        det_id    = det["detection_id"]
        img_idx   = det.get("source_image_index", 0)
        part_name = det.get("location", "unknown")

        img_node  = f"IMG_{img_idx}"
        part_node = f"PART_{part_name}"

        G.add_node(
            img_node,
            node_type = "image",
            index     = img_idx,
            path      = det.get("source_image_path", ""),
        )
        G.add_node(
            part_node,
            node_type = "vehicle_part",
            part_name = part_name,
        )
        G.add_node(det_id, node_type="detection", **det)

        G.add_edge(img_node,  det_id,    relation="contains")
        G.add_edge(det_id,    part_node, relation="located_on")

    return G


def fuse_detections(
    all_detections: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    Fuse a flat list of per-image detections into a deduplicated damage list.

    Algorithm
    ---------
    1. Build a NetworkX DiGraph via build_claims_graph().
    2. For each PART node, gather all DETECTION nodes pointing to it.
    3. Select the Golden Record — detection with highest confidence.
    4. Enrich the Golden Record with multi-image metadata.
    5. Detect image-recycling fraud loops (identical bboxes across images).

    Parameters
    ----------
    all_detections : list of detection dicts from cv_worker_node outputs

    Returns
    -------
    fused        : list of Golden Record dicts (one per unique damage)
    recycling_flags : list of fraud flag strings
    graph_stats  : dict with node/edge counts for the audit trace
    """
    G = build_claims_graph(all_detections)

    n_images = sum(
        1 for _, d in G.nodes(data=True) if d.get("node_type") == "image"
    )
    n_parts  = sum(
        1 for _, d in G.nodes(data=True) if d.get("node_type") == "vehicle_part"
    )

    fused: List[Dict[str, Any]] = []
    recycling_flags: List[str]  = []

    for node, data in G.nodes(data=True):
        if data.get("node_type") != "vehicle_part":
            continue

        part_name = data["part_name"]

        # All DETECTION nodes pointing to this PART
        det_ids = [
            pred for pred in G.predecessors(node)
            if G.nodes[pred].get("node_type") == "detection"
        ]
        if not det_ids:
            continue

        # Golden Record = highest-confidence detection
        golden_id = max(det_ids, key=lambda d: G.nodes[d].get("confidence", 0.0))
        golden    = dict(G.nodes[golden_id])

        # Enrich with multi-image metadata
        seen_indices = sorted(set(
            G.nodes[d].get("source_image_index", 0) for d in det_ids
        ))
        golden["visibility_count"]  = len(det_ids)
        golden["seen_in_indices"]   = seen_indices
        golden["primary_image_idx"] = G.nodes[golden_id].get("source_image_index", 0)
        golden["fused_from_count"]  = len(det_ids)
        golden["is_fused"]          = len(det_ids) > 1

        # ── Image-recycling fraud detection ───────────────────────────────────
        # Identical bounding boxes across multiple images → same photo resubmitted
        if len(det_ids) > 1:
            bboxes = [
                tuple(G.nodes[d].get("bbox", []))
                for d in det_ids
            ]
            if len(set(bboxes)) == 1:
                flag = (
                    f"IMAGE_RECYCLING_LOOP: Part '{part_name}' has identical "
                    f"bboxes across {len(det_ids)} images — likely same photo "
                    "submitted multiple times"
                )
                recycling_flags.append(flag)
                golden["fraud_recycling_flag"] = True
                print(
                    f"   🚨 Recycling loop on '{part_name}' — "
                    f"identical bbox across {len(det_ids)} images"
                )

        fused.append(golden)
        print(
            f"   ✅ PART '{part_name}': {len(det_ids)} detection(s) → "
            f"1 golden record (conf={golden.get('confidence', 0):.2f}, "
            f"seen_in={seen_indices})"
        )

    graph_stats: Dict[str, Any] = {
        "nodes":             G.number_of_nodes(),
        "edges":             G.number_of_edges(),
        "images_processed":  n_images,
        "parts_detected":    n_parts,
        "raw_detections":    len(all_detections),
        "fused_damages":     len(fused),
        "recycling_flags":   len(recycling_flags),
    }

    return fused, recycling_flags, graph_stats
