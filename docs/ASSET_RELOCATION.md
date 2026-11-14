# Loading archived robot scenes after moving assets

Use `real2sim2real.mujoco_assets.load_model(path)` when an archived MJCF or URDF
contains absolute asset paths whose roots have moved. MuJoCo remains an optional
dependency and is imported only when loading a model.

Provide a JSON object mapping absolute old roots to absolute new roots, then set
`R2S2R_ASSET_PATH_MAP` to that file. Alternatively pass `mapping={...}` to
`load_model`. With no map, the function delegates directly to MuJoCo.

```python
from real2sim2real.mujoco_assets import load_model

model = load_model(
    "archive/scene.xml",
    mapping={"/previous/robot-assets": "/current/robot-assets"},
)
```

Only path-valued XML attributes are relocated. Longest-prefix matching respects
directory-component boundaries. Nested includes are resolved recursively and
include cycles are rejected. Original files remain byte-identical. Temporary
siblings preserve each XML document's relative asset base and are removed even
when loading fails. Scene directories therefore need write permission during
loading. The helper does not rewrite trajectories, geometry, contacts or control
parameters, and it does not download missing assets.

When relocating an environment, preserve its interpreter and installed packages
as well as its scene assets. Editable installs and environment shebangs can keep
an old checkout alive even after configuration paths have changed. Verify the
consumer with the old tree unavailable and compare numerical model arrays, not
just successful XML parsing. Unit tests cover mapping boundaries, invalid maps,
nested includes, cleanup and CPU model equivalence.
