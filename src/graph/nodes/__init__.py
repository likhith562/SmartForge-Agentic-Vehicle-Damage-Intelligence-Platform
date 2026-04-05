"""
SmartForge graph nodes sub-package.
Each module contains exactly one LangGraph node function plus its
supporting router (where applicable).

All nodes are exported here for convenient import in workflow.py.
"""

from .intake               import intake_node
from .fraud                import fraud_node, fraud_router
from .human_audit          import human_audit_node
from .map_reduce           import map_images_node, cv_worker_node, fusion_node
from .perception           import perception_node, perception_retry_node
from .gemini_agent         import gemini_agent_node
from .false_positive_gate  import false_positive_gate_node
from .health_monitor       import health_monitor_node, health_monitor_router
from .verification_v2      import verification_v2_node
from .reasoning            import reasoning_node
from .decision             import decision_node
from .report               import report_node

__all__ = [
    # Batch 1: Fraud layer
    "intake_node",
    "fraud_node",
    "fraud_router",
    "human_audit_node",
    # Batch 2: Multi-image map-reduce
    "map_images_node",
    "cv_worker_node",
    "fusion_node",
    # Core CV pipeline
    "perception_node",
    "perception_retry_node",
    "gemini_agent_node",
    "false_positive_gate_node",
    "health_monitor_node",
    "health_monitor_router",
    # Batch 3: Golden Frame
    "verification_v2_node",
    # Reasoning & decision
    "reasoning_node",
    "decision_node",
    "report_node",
]
