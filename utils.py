"""
Utility functions for Lightning nnU-Net wrappers.
"""

import os
from pathlib import Path


def set_nnunet_env(
    nnunet_raw,
    nnunet_preprocessed,
    nnunet_results,
):
    """
    Set nnU-Net v2 environment variables.

    nnU-Net expects these exact env variable names:
        nnUNet_raw
        nnUNet_preprocessed
        nnUNet_results
    """

    nnunet_raw = Path(nnunet_raw).resolve()
    nnunet_preprocessed = Path(nnunet_preprocessed).resolve()
    nnunet_results = Path(nnunet_results).resolve()

    nnunet_raw.mkdir(parents=True, exist_ok=True)
    nnunet_preprocessed.mkdir(parents=True, exist_ok=True)
    nnunet_results.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(nnunet_raw)
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed)
    os.environ["nnUNet_results"] = str(nnunet_results)

    return {
        "nnunet_raw": str(nnunet_raw),
        "nnunet_preprocessed": str(nnunet_preprocessed),
        "nnunet_results": str(nnunet_results),
    }
