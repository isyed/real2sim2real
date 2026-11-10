# Upstream code and assets

The extraction preserves upstream copyright notices and bundled license
texts. New adapters do not relicense the included third-party components.
The top-level [Apache-2.0 license](../LICENSE) covers original project
contributions, including new code and documentation, subject to component-specific
terms. Each independently packaged module includes the same license and a
[NOTICE](../NOTICE). The original combined NVIDIA notice is preserved in
`licenses/UPSTREAM-LICENSE`: its source-code terms are Apache-2.0, while upstream
documentation/skills remain CC-BY-4.0. Changing the project's license file does
not change those inherited terms.
Consult `licenses/` and per-file headers before redistribution. Model weights
are not included and have their own acquisition and use terms.

| Component | Upstream / role |
| --- | --- |
| SAM2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2), segmentation package |
| MoGe | [microsoft/MoGe](https://github.com/microsoft/MoGe), metric depth |
| SAM 3D Objects | [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects), separately installed mesh model |
| SAM 3D Body / MHR | [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body), included model adapters/decoding and external weights; SAM license |
| FoundationPose | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose), included pose tracking implementation |
| SOMA | `py-soma-x` 0.2.1, separately installed human model/inversion |
| Pink / Pinocchio | `pin-pink` / `pin`, constrained robot IK |
| FlashCHORD / FlashSAC | Included NVIDIA Apache-2.0 simulation/training implementation, with generic adapters and hand-closing extension |
| SONIC | External pretrained body-controller assets; not a task-level planner |
| HaMeR | [geopavlakos/hamer](https://github.com/geopavlakos/hamer), separately installed hand-estimation backend; upstream code/model terms apply |
| HaWoR / WiLoR detector | [malik-group/HaWoR](https://github.com/malik-group/HaWoR), pinned external runtime; upstream HaWoR model terms and detector dependencies retain their licenses |
| MANO | Separately supplied licensed hand-model assets; never included in project wheels |

Large robot assets, model binaries, datasets, authentication files and run
histories are deliberately excluded. Public release should include a review
of component/model redistribution terms and the documented GPU acceptance
test, rather than assuming that making the repository public makes every
model freely redistributable.
