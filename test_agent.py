"""
Tests for YOUR bot (agent.py). Run them from the bundle folder:

    python -m pytest -q

They talk to your bot the same way the judge does (through its web routes,
but without starting a real server), so a passing run means the judge can
play with your bot. Add your own tests at the bottom!
"""

import importlib.util
import os
import random
import sys

import pytest
from fastapi.testclient import TestClient

# Works whether this file sits next to agent.py or inside a tests/ folder.
HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = HERE if os.path.isfile(os.path.join(HERE, "game.py")) else os.path.dirname(HERE)
sys.path.insert(0, BUNDLE)  # so agent.py can import game.py and evaluator.py

from game import Action, AgentConfig, Game, make_deal  # noqa: E402
from judge_engine import coerce_action  # noqa: E402  (reads answers exactly like the judge)

TOKEN = "test-token"
AUTH = {"X-Agent-Token": TOKEN}


def load_agent():
    """Load a fresh copy of agent.py (its own MEMORY and globals)."""
    spec = importlib.util.spec_from_file_location(
        "agent_under_test",
        os.path.join(BUNDLE, "agent.py")
    )

    if spec is None or spec.loader is None:
        raise ImportError("Could not load agent.py")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bot():
    """(agent module, test client) with the tournament's token check switched on."""
    module = load_agent()
    setattr(module, "TOKEN", TOKEN)
    return module, TestClient(module.app)


def start_half(client, half=1, seat=0):
    """POST /new-half the way the judge does."""
    body = {"half": half, "seat": seat, "config": AgentConfig().to_dict()}
    return client.post("/new-half", json=body, headers=AUTH)


def play_hands(client, num_hands, seed="test", raise_rate=0.3):
    """Play hands where YOUR bot is seat 0 and a random opponent is seat 1.

    Every decision goes through /act exactly like in a real match, and
    finished hands are delivered with the next /act (like the judge does).
    Returns the list of every action your bot chose.
    """
    rng = random.Random(seed)
    pending = []      # finished hands waiting to be delivered
    chosen = []
    for hand in range(num_hands):
        game = Game(make_deal(seed, hand), button_seat=hand % 2, hand_index=hand)
        while not game.is_terminal:
            seat = game.to_act
            if seat is None:
                raise RuntimeError("Game is not terminal but no player can act")
            obs = game.observation(seat)
            if seat == 0:
                r = client.post("/act", json={"observation": obs.to_dict(), "hand_results": pending},
                                headers=AUTH)
                pending = []
                assert r.status_code == 200, f"/act failed: {r.json()}"
                reply = r.json()
                assert reply.get("hand_end_errors", []) == [], \
                    f"on_hand_end crashed: {reply['hand_end_errors'][0]['traceback']}"
                action = coerce_action(reply["action"])
                assert action in obs.legal_actions, \
                    f"choose_action returned {reply['action']!r}; legal were {[a.name for a in obs.legal_actions]}"
                chosen.append(action)
            else:
                # Random opponent that raises sometimes, so your bot faces bets.
                raises = [a for a in obs.legal_actions if a in (Action.RAISE_HALF, Action.RAISE_POT)]
                action = rng.choice(raises) if raises and rng.random() < raise_rate else Action.CHECK_CALL
            if action is None:
                raise RuntimeError("No action was selected")
            game.apply(action)
        pending.append(game.hand_result(0).to_dict())
    r = client.post("/hand-end", json={"results": pending}, headers=AUTH)  # end of half
    assert r.status_code == 200 and r.json()["hand_end_errors"] == []
    return chosen


# ---------------------------------------------------------------------------
# Your bot plays correctly
# ---------------------------------------------------------------------------

def test_names_are_set(bot):
    _, client = bot
    info = client.get("/").json()
    assert isinstance(info["participant"], str) and info["participant"]
    assert isinstance(info["agent_name"], str) and info["agent_name"]


def test_new_half_works(bot):
    _, client = bot
    r = start_half(client)
    assert r.status_code == 200, f"new_half crashed: {r.json()}"


def test_every_decision_is_legal(bot):
    # 200 hands with a random opponent: every choose_action answer must be legal.
    _, client = bot
    start_half(client)
    play_hands(client, 200)


def test_works_in_both_seats_and_both_halves(bot):
    _, client = bot
    for half, seat in ((1, 0), (2, 1)):
        assert start_half(client, half, seat).status_code == 200
        play_hands(client, 30, seed=f"half{half}")


def test_handles_aggressive_opponent(bot):
    # An opponent that raises whenever it can: your bot must cope with
    # big bets and all-ins.
    _, client = bot
    start_half(client)
    play_hands(client, 100, seed="aggressive", raise_rate=1.0)


def test_memory_is_fresh_each_half(bot):
    module, client = bot
    start_half(client)
    play_hands(client, 10)
    start_half(client, half=2, seat=1)
    after_reset = dict(module.MEMORY)
    fresh = load_agent()
    fresh.new_half(AgentConfig(), fresh.MEMORY)
    assert after_reset.keys() == fresh.MEMORY.keys(), "memory should only hold what new_half() puts there"


# ---------------------------------------------------------------------------
# The server plumbing still works (only fails if you edited it)
# ---------------------------------------------------------------------------

def test_token_is_checked(bot):
    _, client = bot
    body = {"half": 1, "seat": 0, "config": {}}
    assert client.post("/new-half", json=body).status_code == 401
    assert client.post("/new-half", json=body, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_bad_token_is_refused_before_the_body_is_read(bot):
    # Without the right token the bot must answer 401 without parsing the body,
    # so junk sent by anyone else can't slow your bot down.
    _, client = bot
    junk = b"this is not JSON" * 1000
    assert client.post("/act", content=junk).status_code == 401
    assert client.post("/act", content=junk, headers={"X-Agent-Token": "wrong"}).status_code == 401


def test_malformed_body_is_rejected(bot):
    _, client = bot
    start_half(client)
    assert client.post("/act", content=b"not json", headers=AUTH).status_code == 400
    assert client.post("/act", json=[1, 2, 3], headers=AUTH).status_code == 400


def test_act_before_new_half_is_refused(bot):
    _, client = bot
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    assert client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH).status_code == 409


def test_crashes_are_reported_to_the_judge(bot):
    # If choose_action raises, the judge must get a 500 with your traceback.
    module, client = bot

    def broken(obs, memory):
        raise ValueError("oops")

    module.choose_action = broken
    start_half(client)
    obs = Game(make_deal(1, 0), 0).observation(0).to_dict()
    r = client.post("/act", json={"observation": obs, "hand_results": []}, headers=AUTH)
    assert r.status_code == 500
    assert "ValueError: oops" in r.json()["error"] and "Traceback" in r.json()["traceback"]


def test_reads_port_from_environment():
    source = open(os.path.join(BUNDLE, "agent.py"), encoding="utf-8").read()
    assert 'os.environ.get("PORT"' in source, "agent.py must read its port from $PORT"
    assert '0.0.0.0' in source, "agent.py must listen on 0.0.0.0 so the judge can reach it"


# ---------------------------------------------------------------------------
# Add your own tests below, e.g. "with two aces preflop my bot raises".
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

def test_premium_preflop_hand_is_stronger(bot):
    module, _ = bot

    # Card encoding: rank * 4 + suit
    premium = [32, 33, 28, 29, 24]  # AA KK Q
    weak = [0, 5, 10, 15, 20]      # 6 7 8 9 T

    assert module.preflop_strength(premium) > module.preflop_strength(weak)


def test_preflop_strength_stays_in_bounds(bot):
    module, _ = bot
    rng = random.Random(42)

    for _ in range(1000):
        cards = rng.sample(range(36), 5)
        strength = module.preflop_strength(cards)

        assert 0.0 <= strength <= 1.0


def test_equity_is_deterministic(bot):
    module, _ = bot

    game = Game(make_deal(42, 0), button_seat=0)
    seat = game.to_act
    if seat is None:
        raise RuntimeError("Game is not terminal but no player can act")

    obs = game.observation(seat)

    first = module.estimate_equity(obs)
    second = module.estimate_equity(obs)

    assert first == second


def test_simulation_count_is_bounded(bot):
    module, _ = bot

    game = Game(make_deal(42, 0), button_seat=0)
    seat = game.to_act
    if seat is None:
        raise RuntimeError("Game is not terminal but no player can act")

    obs = game.observation(seat)

    count = module.simulation_count(obs)

    assert 0 <= count <= 10


def test_check_is_preferred_when_raising_unavailable(bot):
    module, _ = bot

    game = Game(make_deal(42, 0), button_seat=0)

    seat = game.to_act
    if seat is None:
        raise RuntimeError("Game is not terminal but no player can act")

    obs = game.observation(seat)

    memory = {}
    module.new_half(AgentConfig(), memory)

    # A state with no betting cost and a full raise cap.
    from dataclasses import replace

    restricted = replace(
        obs,
        to_call=0,
        raises_this_street=4,
        legal_actions=(Action.CHECK_CALL,)
    )

    action = module.choose_action(restricted, memory)

    assert action == Action.CHECK_CALL


def test_pot_odds_increase_with_call_cost():
    def pot_odds(pot, to_call):
        return to_call / (pot + to_call)

    assert pot_odds(20, 2) < pot_odds(20, 10)
    assert pot_odds(20, 10) < pot_odds(20, 20)


def test_opponent_aggression_neutral_initially(bot):
    module, _ = bot

    memory = {}
    module.new_half(AgentConfig(), memory)

    assert module.opponent_aggression(memory) == pytest.approx(0.30)


def test_opponent_aggression_increases(bot):
    module, _ = bot

    memory = {}
    module.new_half(AgentConfig(), memory)

    memory["opp_actions"] = 10
    memory["opp_raises"] = 6

    assert module.opponent_aggression(memory) == pytest.approx(0.60)


def test_opponent_aggression_is_capped(bot):
    module, _ = bot

    memory = {}
    module.new_half(AgentConfig(), memory)

    memory["opp_actions"] = 100
    memory["opp_raises"] = 100

    assert module.opponent_aggression(memory) == pytest.approx(0.65)


# PERFORMANCE BENCHMARK
# run with 'python -m pytest -s -q -k strategy_performance'
def test_strategy_performance(bot):
    import time

    module, _ = bot
    memory = {}
    module.new_half(AgentConfig(), memory)

    rng = random.Random(42)
    observations = []

    for hand in range(1000):
        game = Game(
            make_deal(rng.randint(0, 1000000), hand),
            button_seat=hand % 2,
            hand_index=hand
        )

        seat = game.to_act
        assert seat is not None
        observations.append(game.observation(seat))

    start = time.perf_counter()

    for obs in observations:
        action = module.choose_action(obs, memory)
        assert action in obs.legal_actions

    elapsed = time.perf_counter() - start

    print(f"\n1000 decisions: {elapsed:.3f} seconds")
    print(f"Average decision: {elapsed / 1000 * 1000:.3f} ms")

# Gambling Addict v.2 - regression tests for Gambling Addict
def test_opponent_history_counts_multiple_actions(bot):
    from types import SimpleNamespace

    module, _ = bot
    memory = {}
    module.new_half(None, memory)

    def entry(is_me, action):
        return SimpleNamespace(is_me=is_me, action=action)

    obs = SimpleNamespace(
        hand_index=1,
        action_history=[
            entry(False, Action.RAISE_HALF),
            entry(True, Action.CHECK_CALL),
            entry(False, Action.RAISE_POT),
            entry(False, Action.CHECK_CALL),
        ],
    )

    module.record_opponent_action(obs, memory)

    assert memory["opp_actions"] == 3
    assert memory["opp_raises"] == 2
    assert memory["opp_half_raises"] == 1
    assert memory["opp_pot_raises"] == 1
    assert memory["opp_calls_checks"] == 1


def test_opponent_history_does_not_double_count(bot):
    from types import SimpleNamespace

    module, _ = bot
    memory = {}
    module.new_half(None, memory)

    obs = SimpleNamespace(
        hand_index=1,
        action_history=[
            SimpleNamespace(
                is_me=False,
                action=Action.RAISE_POT,
            )
        ],
    )

    module.record_opponent_action(obs, memory)
    module.record_opponent_action(obs, memory)

    assert memory["opp_actions"] == 1
    assert memory["opp_raises"] == 1


def test_opponent_history_resets_each_hand(bot):
    from types import SimpleNamespace

    module, _ = bot
    memory = {}
    module.new_half(None, memory)

    for hand_index in (1, 2):
        obs = SimpleNamespace(
            hand_index=hand_index,
            action_history=[
                SimpleNamespace(
                    is_me=False,
                    action=Action.RAISE_HALF,
                )
            ],
        )

        module.record_opponent_action(obs, memory)

    assert memory["opp_actions"] == 2
    assert memory["opp_raises"] == 2


