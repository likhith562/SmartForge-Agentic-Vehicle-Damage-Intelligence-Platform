"""
SmartForge — intake_node
=========================
First node in the LangGraph pipeline.  Validates and preprocesses the
submitted image, then writes the results into state so every downstream
node receives clean, consistently-sized arrays.

Responsibilities
----------------
1. File existence and decodability check  — raises RuntimeError on failure
   so LangGraph surfaces the error immediately rather than producing silent
   bad state.

2. Adaptive image condition analysis  (via cv.perception.analyse_image_conditions)
   - Computes HSV V-channel variance to detect shiny/reflective surfaces
   - Sets ``adaptive_sahi_conf`` in state (read by perception_node)
   - Sets ``scene_type`` for the audit trace

3. Adaptive resize
   - Upscale  : images smaller than 224px on either dimension are upscaled
     to 640px short-side so SAHI has enough pixels to slice.
   - Downsample: images larger than 4096px on either dimension are reduced
     unless ``skip_downsampling=True`` (vehicle is small in frame — preserving
     pixel density matters more than staying under 4096px).

4. Writes ``image_bgr``, ``image_rgb``, ``image_path`` into state.
   ``image_bgr`` is freed (set to None) inside perception_node once SAM
   and MiDaS have consumed ``image_rgb`` to avoid holding a ~28 MB array
   through the rest of the graph.

State mutations returned
------------------------
    image_path          str   — possibly updated with "_resized" suffix
    image_bgr           np.ndarray
    image_rgb           np.ndarray
    adaptive_sahi_conf  float
    scene_type          str
    pipeline_trace      dict  — "intake_agent" entry added
    messages            list  — one entry appended
"""

import os

import cv2

from src.cv.perception import analyse_image_conditions
from src.graph.state import SmartForgeState, log_msg


def intake_node(state: SmartForgeState) -> dict:
    """
    LangGraph node: validate and preprocess the submitted vehicle image.

    Parameters
    ----------
    state : SmartForgeState
        Expects ``image_path`` to be a valid filesystem path.

    Returns
    -------
    dict — partial state update merged by LangGraph into the running state.

    Raises
    ------
    RuntimeError
        When the file does not exist or OpenCV cannot decode it.
        LangGraph will propagate this as a graph-level error.
    """
    from datetime import datetime, timezone

    path = state["image_path"]
    print(f"\n🔵 [intake] validating → {path}")

    # ── 1. File existence & decodability ─────────────────────────────────────
    if not os.path.exists(path):
        raise RuntimeError(f"intake_node: File not found → {path}")

    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"intake_node: Cannot decode image → {path}")

    h, w = image.shape[:2]

    # ── 2. Adaptive image condition analysis (must run before any resize) ─────
    from src.config.settings import cfg
    conditions = analyse_image_conditions(image, sahi_slice_size=cfg.SAHI_SLICE_SIZE)
    print(
        f"   Scene: {conditions['scene_type']} "
        f"(V-variance={conditions['v_variance']}) "
        f"→ SAHI conf={conditions['adaptive_sahi_conf']}"
    )
    if conditions["skip_downsampling"]:
        print("   ⚠️  Downsampling skipped — detail preservation mode active")

    # ── 3. Adaptive resize ────────────────────────────────────────────────────
    action = "none"

    # Upscale: too small for SAHI to produce meaningful tiles
    if h < 224 or w < 224:
        scale = 640 / min(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        base, ext = os.path.splitext(path)
        path      = f"{base}_resized{ext}"
        cv2.imwrite(path, image)
        h, w   = image.shape[:2]
        action = "upscaled"

    # Downsample: excessively large image that won't harm scratch detection
    if (h > 4096 or w > 4096) and not conditions["skip_downsampling"]:
        scale = 4096 / max(h, w)
        image = cv2.resize(
            image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        base, ext = os.path.splitext(path)
        path      = f"{base}_resized{ext}"
        cv2.imwrite(path, image)
        h, w   = image.shape[:2]
        action = "downsampled"
    elif (h > 4096 or w > 4096) and conditions["skip_downsampling"]:
        action = "downsampling_skipped_detail_preservation"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── 4. Build audit trace entry ────────────────────────────────────────────
    trace_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"Image {w}×{h}px. "
            f"Scene: {conditions['scene_type']} "
            f"(V-variance={conditions['v_variance']}). "
            f"Adaptive SAHI conf={conditions['adaptive_sahi_conf']}. "
            f"Resize action: {action}. "
            + conditions["reasoning"]
        ),
        "decision": "Image accepted — routing to fraud gate.",
        "details": {
            "path":               path,
            "dimensions":         f"{w}×{h}",
            "action":             action,
            "scene_type":         conditions["scene_type"],
            "v_variance":         conditions["v_variance"],
            "adaptive_sahi_conf": conditions["adaptive_sahi_conf"],
            "skip_downsampling":  conditions["skip_downsampling"],
        },
    }

    print(
        f"✅ intake: [{w}×{h}] action={action} | "
        f"scene={conditions['scene_type']} | "
        f"sahi_conf={conditions['adaptive_sahi_conf']}"
    )

    return {
        "image_path":         path,
        "image_bgr":          image,
        "image_rgb":          image_rgb,
        "adaptive_sahi_conf": conditions["adaptive_sahi_conf"],
        "scene_type":         conditions["scene_type"],
        "pipeline_trace": {
            **state["pipeline_trace"],
            "intake_agent": trace_entry,
        },
        "messages": [
            log_msg(
                "intake_agent",
                f"Image [{w}×{h}] scene={conditions['scene_type']} "
                f"sahi_conf={conditions['adaptive_sahi_conf']} "
                f"action={action}.",
            )
        ],
    }
