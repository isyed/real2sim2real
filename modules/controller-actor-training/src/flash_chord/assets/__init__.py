# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
ASSETS_DIR=Path(os.environ.get("R2S2R_ASSETS", str(Path.home()/".cache/real2sim2real/assets")))
URDF_DIR=ASSETS_DIR/"urdfs"
