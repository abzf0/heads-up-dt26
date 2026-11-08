# Initial v.5 smoke evaluation (not a trained result)

- Judge seed 201, 1,000 duplicate deals per matchup.
- v.5 default policy vs archived v3c: -0.4145 bb/deal, CHOP.
- Frozen v.4 baseline vs archived v3c: -0.4355 bb/deal, CHOP.
- Original uploaded archive also had 26 v.4 vs counter matches, average -0.17246 bb/deal (separate result set).

These files prove the training harness can run official judge matches; the small improvement on one seed does not prove v.5 is better. There is deliberately **no** `training/models/best.json` included; run `python3 training/train.py --iterations ...` to generate one with real optimization. Test on completely different seeds before deployment.

The archived v3c source uses an agent label with its words in a different order from the recorded 26-match logs; do not automatically assume an identical build without checking the code revision.
