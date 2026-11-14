"""Run from installed wheels, inside an outer sandbox with no checkout mounts."""

import importlib.util
import json
import sys
from pathlib import Path

from real2sim2real.agentic_policy.campaign import advance, initialize, status, submit

from real2sim2real.artifacts import write_json


def main(root):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    assert "/site-packages/" in str(
        __import__("real2sim2real.agentic_policy", fromlist=[""]).__file__
    )
    for name in ["gap", "gap_core", "asl", "robosimstudio"]:
        assert importlib.util.find_spec(name) is None, name
    assert not Path("/home").exists()
    bundle = root / "fixture"
    (bundle / "author").mkdir(parents=True)
    author = """from pathlib import Path
import json
import importlib.util
assert not Path('/home').exists()
assert importlib.util.find_spec('asl') is None
tools=json.loads(Path('tools.json').read_text())
assert 'geometry.pose_distance' in tools and 'diagnostic.advance' in tools
access=json.loads(Path('CODE_ACCESS.json').read_text())
skills=json.loads(Path('skills.json').read_text())
assert len(skills['sources'])==33
assert Path(skills['adaptation_guide']).is_file()
assert all(s['status']=='source_reference_requires_adaptation' for s in skills['sources'])
assert all(s['status']=='backend_unavailable' for s in skills['adapted_operations'])
import hashlib
manifest=json.loads(Path(skills['manifest']).read_text())
for name, expected in manifest['files'].items():
    assert hashlib.sha256((Path(skills['manifest']).parent/name).read_bytes()).hexdigest()==expected
assert skills==access['skills']
assert access['python_sources'] and all(Path(p).is_file() for p in access['python_sources'])
import zipfile
with zipfile.ZipFile(access['project_source_archive']) as archive:
    assert any(n.startswith('modules/reconstruction/src/') for n in archive.namelist())
    assert archive.read('src/real2sim2real/cli.py')
feedback = json.loads(Path('feedback.json').read_text()) if Path('feedback.json').exists() else None
amount = 2 if feedback else 1
policy = {"version":3,"nodes":{"act":{"type":"tool","tool":"diagnostic.advance","inputs":{"amount":amount}},"done":{"type":"end","status":"success"}},"edges":[["START","act"],["act","done"]],"conditional_edges":{},"subgraphs":{}}
Path('candidate/workflow.json').write_text(json.dumps(policy))
"""
    (bundle / "author/fixture_author.py").write_text(author)
    config = {
        "mode": "automatic",
        "feedback_revisions": 6,
        "instruction": "Diagnostic counter target is 2.",
        "rollout_timeout_s": 20,
        "backend": {"kind": "diagnostic_counter"},
        "author": {"argv": [sys.executable, "fixture_author.py"], "timeout_s": 20},
    }
    write_json(bundle / "experiment.json", config)
    campaign = root / "automatic"
    initialize(bundle, campaign)
    result = advance(campaign)
    assert len(result["rounds"]) == 7
    assert not result["rounds"][0]["result"]["verdict"]["task_completed"]
    assert all(r["result"]["verdict"]["task_completed"] for r in result["rounds"][1:])
    assert advance(campaign) == result
    # Manual workflow uses the same evaluation path and physical verdict fields.
    config.update(mode="manual", feedback_revisions=1)
    write_json(bundle / "experiment.json", config)
    initialize(bundle, root / "manual")
    assert advance(root / "manual")["next"] == "submit proposal"
    for i in range(2):
        row = submit(root / "manual", campaign / "rounds" / f"P{i}" / "candidate")
        assert row["result"]["verdict"] == result["rounds"][i]["result"]["verdict"]
    config.update(mode="review", feedback_revisions=0)
    write_json(bundle / "experiment.json", config)
    initialize(bundle, root / "review")
    gate = advance(root / "review")
    assert status(root / "review")["completed_proposals"] == 0
    submit(root / "review", gate["awaiting_human_review"])
    assert status(root / "review")["completed_proposals"] == 1
    # Script policies can use installed code and tools but cannot see the host.
    scripted = root / "scripted-candidate"
    scripted.mkdir()
    (scripted / "policy.py").write_text("""from pathlib import Path
from typing import TypedDict
from real2sim2real.agentic_policy.api import NodeContext
class Result(TypedDict):
    value: int
def run(ctx: NodeContext) -> Result:
    assert not Path('/home').exists()
    assert ctx.tool('geometry.pose_distance', a=[0.,0.,0.], b=[1.,0.,0.])['distance']==1.
    return ctx.tool('diagnostic.advance', amount=2)
""")
    write_json(
        scripted / "workflow.json",
        {
            "version": 3,
            "nodes": {
                "act": {"type": "script", "script": "policy.py", "inputs": {}},
                "done": {"type": "end", "status": "success"},
            },
            "edges": [["START", "act"], ["act", "done"]],
            "conditional_edges": {},
            "subgraphs": {},
        },
    )
    config.update(
        mode="manual",
        feedback_revisions=0,
        evaluation={
            "target": "rl",
            "robustness_cases": [
                {"name": "repeat", "backend": {"kind": "diagnostic_counter"}}
            ],
            "admission": {"rl": {"limits": {"value": {"min": 2, "max": 2}}}},
        },
    )
    write_json(bundle / "experiment.json", config)
    initialize(bundle, root / "scripted")
    checked = submit(root / "scripted", scripted)
    assert checked["result"]["assessment"]["admission"]["rl"]["admitted"], checked
    assert len(checked["result"]["robustness_cases"]) == 1
    # Frozen input tampering must fail before any new run.
    (root / "manual/bundle/author/tamper.txt").write_text("changed")
    try:
        status(root / "manual")
    except ValueError:
        pass
    else:
        raise AssertionError("Frozen bundle mutation accepted")
    receipt = {
        "passed": True,
        "automated_proposals": 7,
        "feedback_revisions": 6,
        "manual_verdict_parity": True,
        "review_gate": True,
        "script_rpc_and_isolation": True,
        "complete_author_tool_catalog_and_source_access": True,
        "all_33_upstream_skill_sources_and_hashes_readable": True,
        "unavailable_backend_adaptations_not_advertised_executable": True,
        "staged_robustness_and_declared_rl_criteria": True,
        "resume_no_duplicate_rollouts": True,
        "frozen_inputs_checked": True,
        "neighboring_checkouts_visible": False,
        "upstream_distributions_installed": False,
        "external_author_calls": False,
        "scope": "Orchestration fixture. Separate MuJoCo acceptance required.",
    }
    write_json(root / "acceptance.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
