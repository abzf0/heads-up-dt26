Team Name: Internship Seekers
Agent Name: Gambling Addict

README.md written by Andrew Feng (by hand! Through typing!)

Important commands for testing:
Testing seeds 1-100:
  mkdir -p benchmarks/v#_vs_v#

  for seed in {1..100}; do
    echo "===== SEED $seed / 100 ====="

    python3 judge_engine.py \
      --agent1 50## \
      --agent2 50## \
      --deals 1000 \
      --seed "$seed" \
      --json "benchmarks/v#_vs_v#/seed_${seed}.json"
  done

To save benchmark data to .json file:
python3 average_benchmarks.py benchmarks/v#_vs_v# \
  --pattern "seed_*.json" \
  --save
  


Agent Evolution:
v.1: 

