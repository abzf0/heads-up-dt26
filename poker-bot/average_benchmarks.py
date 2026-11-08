"""
python3 average_benchmarks.py benchmarks/[FOLDER_NAME] \
  --pattern "seed_*.json" \
  --save
"""

#!/usr/bin/env python3
"""Summarize existing judge_engine.py match-result JSON files; does not run matches."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Average existing poker judge JSON results.')
    parser.add_argument('path', nargs='?', default='benchmarks', help='Directory containing match JSON files')
    parser.add_argument('--pattern', default='v3_vs_v1_seed*.json', help='Filename glob within the directory')
    parser.add_argument('--save', action='store_true', help='Write summary.json alongside the input files')
    args = parser.parse_args()

    directory = Path(args.path)
    paths = sorted(directory.glob(args.pattern))
    records = []
    for path in paths:
        try:
            with path.open(encoding='utf-8') as f:
                item = json.load(f)
            if item.get('kind') != 'match_result':
                continue
            n = int(item['num_deals'])
            mean = float(item['mean_bb_per_deal'])
            if n <= 0 or not math.isfinite(mean):
                raise ValueError('invalid num_deals or mean')
            records.append((path, item, n, mean))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f'Skipping {path}: {exc}')

    if not records:
        parser.error(f'No valid match results found in {directory / args.pattern}')

    total_deals = sum(r[2] for r in records)
    weighted_mean = sum(r[2] * r[3] for r in records) / total_deals
    means = sorted(r[3] for r in records)
    mid = len(means) // 2
    median = means[mid] if len(means) % 2 else (means[mid-1] + means[mid]) / 2
    verdicts = Counter(r[1].get('verdict', 'UNKNOWN') for r in records)

    # Combine per-deal sample variances, if each judge JSON includes std_bb.
    ci = None
    if total_deals > 1 and all('std_bb' in r[1] for r in records):
        ss = sum((r[2] - 1) * float(r[1]['std_bb'])**2 + r[2] * (r[3] - weighted_mean)**2 for r in records)
        se = math.sqrt(ss / (total_deals - 1) / total_deals)
        ci = [weighted_mean - 1.96 * se, weighted_mean + 1.96 * se]

    result = {
        'matches': len(records), 'total_duplicate_deals': total_deals,
        'weighted_mean_bb_per_deal': weighted_mean, 'median_match_bb_per_deal': median,
        'best_match_bb_per_deal': max(means), 'worst_match_bb_per_deal': min(means),
        'verdicts': dict(verdicts), 'pooled_95pct_ci_bb_per_deal': ci,
    }
    print(f'Matches:              {result["matches"]}')
    print(f'Total duplicate deals: {total_deals}')
    print(f'Average bb/deal:       {weighted_mean:+.4f}')
    print(f'Median match bb/deal:  {median:+.4f}')
    print(f'Best / worst match:   {max(means):+.4f} / {min(means):+.4f}')
    print(f'Verdicts:             {dict(verdicts)}')
    if ci:
        print(f'Pooled 95% CI:        [{ci[0]:+.4f}, {ci[1]:+.4f}]')
    if args.save:
        target = directory / 'summary.json'
        target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'Saved:                {target}')


if __name__ == '__main__':
    main()
