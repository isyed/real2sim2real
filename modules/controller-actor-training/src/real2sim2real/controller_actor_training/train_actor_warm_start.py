"""Train a new task using actor-only initialization and a fresh learner."""
import argparse
import json
import sys
from pathlib import Path
from .actor_warm_start import training_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor', required=True)
    parser.add_argument('--actor-sha256', required=True)
    parser.add_argument('--transfer-evidence', required=True)
    args, hydra_args = parser.parse_known_args()
    evidence = json.loads(Path(args.transfer_evidence).read_text())
    sys.argv = [sys.argv[0], *hydra_args]
    with training_context(args.actor, expected_sha256=args.actor_sha256, evidence=evidence):
        from .train import main as train
        train()


if __name__ == '__main__':
    main()
