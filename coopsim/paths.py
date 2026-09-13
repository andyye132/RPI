"""Filesystem roots, overridable by environment so the package runs off-site.

AETHER_SCRATCH  bulk episode/npz output (default: $SCRATCH/netcomm_v3, else ./scratch)
AETHER_REPO     repo root, for logs and figures (default: this file's parent)
AETHER_RTI      Utah SPAN radio-tomography dataset root (default: $AETHER_SCRATCH/rti_span)
"""
import os
from pathlib import Path

REPO = Path(os.environ.get("AETHER_REPO", Path(__file__).resolve().parent.parent))
SCRATCH = Path(os.environ.get(
    "AETHER_SCRATCH",
    Path(os.environ.get("SCRATCH", REPO / "scratch")) / "netcomm_v3"))

RUNS = SCRATCH / "runs"
LOGS = Path(os.environ.get("AETHER_LOGS", REPO / "results" / "logs"))
FIGS = Path(os.environ.get("AETHER_FIGS", REPO / "netcomm_figs" / "v3"))
RTI = Path(os.environ.get("AETHER_RTI", SCRATCH.parent / "rti_span"))
