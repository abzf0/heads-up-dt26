"""Store measured poker match results and compare agent versions.

Does not run the judge; enter values from its completed match report.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / 'benchmarks' / 'results.json'


def load_results():
    if not RESULTS.exists():
        return []
    data = json.loads(RESULTS.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('results.json must contain a JSON list')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--agent', required=True, help='e.g. baseline-v1 or equity-v2')
    p.add_argument('--opponent', required=True)
    p.add_argument('--wins', type=int, required=True, help='Match/deal units; use the same unit for all comparisons')
    p.add_argument('--losses', type=int, required=True)
    p.add_argument('--ties', type=int, default=0)
    p.add_argument('--time-seconds', type=float, help='Actual judge-reported time spent by agent')
    p.add_argument('--errors', type=int, default=0)
    p.add_argument('--seed', type=int)
    p.add_argument('--notes', default='')
    args = p.parse_args()
    if min(args.wins, args.losses, args.ties, args.errors) < 0:
        p.error('counts cannot be negative')
    if args.time_seconds is not None and args.time_seconds < 0:
        p.error('time-seconds cannot be negative')
    hands = args.wins + args.losses + args.ties
    if hands == 0:
        p.error('at least one result is required')
    entry = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'agent': args.agent,
        'opponent': args.opponent,
        'wins': args.wins,
        'losses': args.losses,
        'ties': args.ties,
        'units': hands,
        'score_rate': round((args.wins + args.ties * 0.5) / hands, 6),
        'time_seconds': args.time_seconds,
        'errors': args.errors,
        'seed': args.seed,
        'notes': args.notes,
    }
    results = load_results()
    results.append(entry)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(f"Saved to {RESULTS}: {args.agent} vs {args.opponent}: {entry['score_rate']:.1%} score rate")


if __name__ == '__main__':
    main()
