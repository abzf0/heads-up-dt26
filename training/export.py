#!/usr/bin/env python3
"""Create a submission-ready agent.py from v5 and the selected training params.

Nothing is changed in-place. Source bot is backed up/untouched; final bot is
written to --output for review, followed by python test_agent.py / smoke tests.
"""
import argparse
import json
import re
from pathlib import Path
from train import ROOT, DEFAULT, read_params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT / 'training/models/best.json')
    ap.add_argument('--output', type=Path, default=ROOT / 'agent_v5_submission.py')
    args = ap.parse_args()
    model = read_params(args.model)
    source = (ROOT / 'agent.py').read_text()
    marker = re.compile(r'(?ms)^_DEFAULT_V5_PARAMS = \{.*?^\}\n')
    matches = list(marker.finditer(source))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one default parameter block')
    indented = ',\n'.join(f'    "{key}": {model[key]!r}' for key in DEFAULT)
    payload = '_DEFAULT_V5_PARAMS = {\n' + indented + '\n}\n'
    updated = marker.sub(lambda _: payload, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(updated)
    print(f'Wrote: {args.output}')
    print('To submit, copy this file to agent.py, then rerun tests and commit.')


if __name__ == '__main__':
    main()
