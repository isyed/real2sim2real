# Local translation repair and independent checks

When size, orientation and unaffected frames have credible evidence, bounded
translation-only windows can test motion defects without refitting everything.
Pin source geometry, masks, predicted depth, calibration and camera identities;
freeze source code and keep the parent. Recompute a fresh calibration receipt if
the current module cannot reproduce an old report; never rewrite that report.

Preserve declared boundary poses and all rotations. Check fitting and held-out
views separately, then whole-clip seams. Held-out development frames are not an
unseen test set once they have guided trial selection. A good mask silhouette
does not establish valid depth, human alignment or executable motion.

A bounded downstream trial reduced a local maximum step from 8.27 to 2.04 cm
and improved held-out mask IoU from 0.9517 to 0.9633. Its depth and image gates
passed, but one reconstructed wrist's distance to the surface increased by
3.90 cm, exceeding the unchanged 5 mm separation-preservation guard. Another
window reduced a 19.50 cm step to 5.40 cm while failing depth at two frames and
increasing wrist separation by 3.74 cm. Neither proposal was selected. Better
averages and smoothness cannot override these paired failures.

The optimizer's own `accepted` field describes its declared geometry gates.
Additional interaction checks belong to the final decision. Use
`reconstruction_repair.decision.combine_decision` to record those checks and
prevent an optimizer-only success from being presented as the final acceptance.
The helper preserves primary rejections, rejects unknown check outcomes, and
leaves full-reference, physics and actor approvals false even on local success.
An independent check supplies its measured value, threshold, scope and evidence;
this composition helper does not generate those measurements or contact labels.

These trials suggest checking human/object consistency before more smoothing.
Predicted wrist proximity is a geometric proxy, not physical contact. If a hand
should be exempt because it is not interacting, establish that phase from
independent observations; do not invent it to bypass a failed guard. No default
repair gate, mesh scale, contact timing or actor-training approval changes here.
