#!/usr/bin/env python3
"""Black-box v5 poker policy optimization using the official local judge.

No changes are made to game.py, evaluator.py, judge_engine.py, or v4/v3c.
Server processes are launched locally, and killed when their evaluation ends.
Keep training seeds separate from final evaluation seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "training" / "models"
RUNS = ROOT / "training" / "results"

OPPONENTS = {
    "v3c": ROOT / "archive" / "agent_v3c.py",
    "v4": ROOT / "archive" / "agent_v4.py",
    "v3": ROOT / "archive" / "agent_v3.py",
    "random": ROOT / "baseline-bots" / "baseline_random.py",
    "calling": ROOT / "baseline-bots" / "baseline_calling_station.py",
    "half": ROOT / "baseline-bots" / "baseline_always_half.py",
    "pot": ROOT / "baseline-bots" / "baseline_always_pot.py",
}

# Match exactly _DEFAULT_V5_PARAMS and _PARAM_BOUNDS in agent.py.
DEFAULT = {
    "preflop_raise_shift": 0.0,
    "passive_exploit_scale": 1.0,
    "postflop_value_shift": 0.0,
    "facing_value_shift": 0.0,
    "call_margin_shift": 0.0,
    "tight_raise_penalty_scale": 1.0,
    "bluff_frequency_scale": 1.0,
    "bluff_fold_gate_shift": 0.0,
    "adapt_fold_pressure": 0.6,
    "adapt_check_defense": 0.7,
}
BOUNDS = {
    "preflop_raise_shift": (-0.12, 0.12),
    "passive_exploit_scale": (0.0, 2.5),
    "postflop_value_shift": (-0.13, 0.13),
    "facing_value_shift": (-0.12, 0.12),
    "call_margin_shift": (-0.08, 0.08),
    "tight_raise_penalty_scale": (0.0, 2.0),
    "bluff_frequency_scale": (0.0, 3.0),
    "bluff_fold_gate_shift": (-0.20, 0.15),
    "adapt_fold_pressure": (0.0, 2.0),
    "adapt_check_defense": (0.0, 2.5),
}
SCALES = {
    "preflop_raise_shift": 0.04,
    "passive_exploit_scale": 0.5,
    "postflop_value_shift": 0.05,
    "facing_value_shift": 0.05,
    "call_margin_shift": 0.03,
    "tight_raise_penalty_scale": 0.45,
    "bluff_frequency_scale": 0.6,
    "bluff_fold_gate_shift": 0.06,
    "adapt_fold_pressure": 0.45,
    "adapt_check_defense": 0.55,
}


def validate_params(data: dict) -> dict:
    if set(data) != set(DEFAULT):
        raise ValueError(f"parameters must have exactly these keys: {sorted(DEFAULT)}")
    output = {}
    for k, v in data.items():
        x = float(v)
        low, high = BOUNDS[k]
        if not math.isfinite(x) or not low <= x <= high:
            raise ValueError(f"{k}={x} outside [{low}, {high}]")
        output[k] = x
    return output


def read_params(path: Path) -> dict:
    data = json.loads(path.read_text())
    return validate_params(data.get("params", data))


def new_candidate(current: dict, rng: random.Random, width: float) -> dict:
    candidate = dict(current)
    count = rng.choice([1, 1, 1, 2, 3])
    for k in rng.sample(list(DEFAULT), count):
        low, high = BOUNDS[k]
        candidate[k] = round(min(high, max(low, current[k] + rng.gauss(0, SCALES[k] * width))), 5)
    return candidate


def available_port(port: int):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"port {port} is occupied: {exc}") from exc


def wait_for_server(port: int, process: subprocess.Popen, seconds: float = 18):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"bot on port {port} stopped (exit={process.returncode})")
        try:
            with urlopen(f"http://127.0.0.1:{port}/", timeout=0.5) as response:
                info = json.load(response)
                if info.get("agent_name"):
                    return info["agent_name"]
        except (OSError, ValueError, TimeoutError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"bot on port {port} did not start within {seconds}s")


@contextmanager
def bot(script: Path, port: int, log: Path, params: dict | None = None):
    available_port(port)
    env = dict(os.environ)
    env["PORT"] = str(port)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("P5_PARAMS_JSON", None)
    if params is not None:
        env["P5_PARAMS_JSON"] = json.dumps(params, separators=(",", ":"))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as outfile:
        p = subprocess.Popen([sys.executable, str(script)], cwd=ROOT, env=env,
                             stdout=outfile, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            name = wait_for_server(port, p)
            yield name
        finally:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=4)


def run_judge(hero_port: int, opp_port: int, deals: int, seed: int, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ROOT / "judge_engine.py"), "--agent1", str(hero_port),
           "--agent2", str(opp_port), "--deals", str(deals), "--seed", str(seed),
           "--json", str(path)]
    job = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         timeout=max(80, int(0.10 * deals + 45)))
    if job.returncode != 0 or not path.is_file():
        raise RuntimeError(f"judge failed: {job.returncode}\n{job.stdout[-1600:]}\n{job.stderr[-1600:]}")
    result = json.loads(path.read_text())
    if result.get("kind") != "match_result" or result.get("num_deals") != deals:
        raise RuntimeError(f"invalid result {path}")
    for agent in ("agent1", "agent2"):
        for half in result["agent_stats"][agent]:
            if half.get("forfeited") or half.get("timed_out") or half.get("errors"):
                raise RuntimeError(f"judge fault for {agent}, seed={seed}, result={path}")
    return result


def evaluate(params: dict, opponents: list[str], seeds: list[int], deals: int,
             outdir: Path, hero_port: int, opp_port: int, hero_file: str = "agent.py") -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    per_opponent = {}
    with bot(ROOT / hero_file, hero_port, outdir / "hero.log",
             params if hero_file == "agent.py" else None) as hero_name:
        for opponent in opponents:
            results = []
            with bot(OPPONENTS[opponent], opp_port, outdir / f"opponent_{opponent}.log") as opp_name:
                print(f"  {hero_name} vs {opp_name}", flush=True)
                for seed in seeds:
                    path = outdir / opponent / f"seed_{seed}.json"
                    data = run_judge(hero_port, opp_port, deals, seed, path)
                    results.append(data)
                    print(f"    seed={seed} {data['mean_bb_per_deal']:+.4f} bb/deal, {data['verdict']}", flush=True)
            average = sum(x["mean_bb_per_deal"] * x["num_deals"] for x in results) / sum(x["num_deals"] for x in results)
            per_opponent[opponent] = {
                "bb_per_deal": round(average, 6),
                "matches": len(results),
                "deals": deals * len(results),
                "verdicts": {v: sum(x["verdict"] == v for x in results) for v in {x["verdict"] for x in results}},
            }
    edges = [x["bb_per_deal"] for x in per_opponent.values()]
    # Returns above +2 are deliberately capped for selection. A +10 edge
    # against random must not disguise a loss against an adaptive opponent;
    # the real tournament gives at most one match point for each matchup.
    capped = [min(2.0, max(-4.0, x)) for x in edges]
    fitness = 0.40 * (sum(capped) / len(capped)) + 0.60 * min(capped)
    report = {
        "hero": hero_file,
        "params": params if hero_file == "agent.py" else None,
        "opponents": per_opponent,
        "deals_per_seed": deals,
        "seeds": seeds,
        "fitness": round(fitness, 6),
        "min_opponent_bb_per_deal": min(edges),
        "mean_opponent_bb_per_deal": round(sum(edges) / len(edges), 6),
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=8, help="number of new proposals; 0 = evaluate only")
    parser.add_argument("--deals", type=int, default=200, help="duplicate deals per match (final: 1000)")
    parser.add_argument("--seeds", default="1,2", help="fixed seeds, comma separated")
    parser.add_argument("--opponents", default="v3c,v4,random,calling,half,pot",
                        help=f"comma separated ({','.join(OPPONENTS)})")
    parser.add_argument("--start", type=Path, help="best.json or JSON parameter file")
    parser.add_argument("--bot", choices=("v5", "v4"), default="v5",
                        help="v4 only available for baseline evaluation (--iterations 0)")
    parser.add_argument("--hero-port", type=int, default=5610)
    parser.add_argument("--opp-port", type=int, default=5611)
    parser.add_argument("--rng-seed", type=int, default=54321)
    parser.add_argument("--output", type=Path, help="custom run directory; default training/results/run_...")
    args = parser.parse_args()
    if args.iterations < 0 or args.deals < 1:
        parser.error("iterations must be >=0 and deals >=1")
    if args.hero_port == args.opp_port:
        parser.error("hero/opp ports must differ")
    if args.bot == "v4" and args.iterations:
        parser.error("v4 is evaluation-only; use --iterations 0")
    opponents = list(dict.fromkeys(x.strip() for x in args.opponents.split(",") if x.strip()))
    if not opponents or any(x not in OPPONENTS for x in opponents):
        parser.error(f"opponents must be from: {', '.join(OPPONENTS)}")
    try:
        seeds = [int(x.strip()) for x in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if not seeds or len(seeds) != len(set(seeds)):
        parser.error("seeds must be nonempty and unique")
    if args.bot == "v4":
        initial = dict(DEFAULT)
    else:
        initial = read_params(args.start) if args.start else dict(DEFAULT)
    rng = random.Random(args.rng_seed)
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    out = args.output or RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    count = args.iterations + 1
    best = dict(initial)
    best_fitness = -float("inf")
    for index in range(count):
        width = max(0.22, 1.0 - index / max(1, 1.25 * count))
        proposal = (dict(initial) if index == 0 else new_candidate(best, rng, width))
        print(f"\nCandidate {index}/{args.iterations}: {proposal}", flush=True)
        folder = out / f"candidate_{index:03d}"
        report = evaluate(proposal, opponents, seeds, args.deals, folder,
                          args.hero_port, args.opp_port,
                          "archive/agent_v4.py" if args.bot == "v4" else "agent.py")
        report["candidate_number"] = index
        reports.append(report)
        print(f"  fitness={report['fitness']:+.5f}, worst={report['min_opponent_bb_per_deal']:+.5f}", flush=True)
        if report["fitness"] > best_fitness:
            best_fitness = report["fitness"]
            best = dict(proposal)
            if args.bot == "v5" and args.iterations > 0:
                MODELS.mkdir(parents=True, exist_ok=True)
                (MODELS / "best.json").write_text(json.dumps({"params": best, "fitness": best_fitness,
                  "trained_opponents": opponents, "training_seeds": seeds,
                  "deals_per_seed": args.deals,
                  "warning": "Requires independent validation before deployment."}, indent=2) + "\n")
        (out / "search_results.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(f"\nCOMPLETE, best={best_fitness:+.5f}, output={out}", flush=True)
    if args.bot == "v5" and args.iterations > 0:
        print(f"Candidate parameters: {MODELS / 'best.json'}", flush=True)


if __name__ == "__main__":
    main()
