"""
agent.py - YOUR poker bot. This one file is a complete web server.

Strategy: counter-strategy for v.3 with Monte Carlo equity and opponent adaptation.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import sys

# Load shared tournament modules from the project root, one level above archive/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import traceback
import random
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from collections import Counter

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Internship Seekers"
AGENT_NAME = "Gambling Addict v.3 Counter"

# ===== YOUR CODE HERE =====

# Small sample counts are deliberate. The time limit is strict.
_MC_TRIALS = {
    3: 12,   # flop
    4: 18,   # turn
    5: 26,   # river
}
def new_half(config, memory):
    memory.clear()
    memory.update({
        # Incremental action-history processing
        "history_hand": -1,
        "history_length": 0,
        # Opponent statistics
        "opp_actions": 0,
        "opp_raises": 0,
        "opp_half_raises": 0,
        "opp_pot_raises": 0,
        "opp_checks": 0,
        "opp_calls": 0,
        # Our pressure results
        "pressure_hand": -1,
        "pressured_hands": 0,
        "pressure_folds": 0,
        # General results
        "hands": 0,
        "net": 0,
    })
def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))
def _fallback(obs):
    if Action.CHECK_CALL in obs.legal_actions:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return obs.legal_actions[0]
def _rng_for(obs, extra=0):
    """
    Reproducible mixing without using Python's randomized hash().
    """
    seed = 2166136261
    values = (
        list(obs.hole_cards)
        + list(obs.board)
        + [
            int(obs.hand_index),
            int(obs.pot),
            int(obs.to_call),
            int(extra),
        ]
    )
    for value in values:
        seed ^= int(value) + 31
        seed = (seed * 16777619) & 0xFFFFFFFF
    return random.Random(seed)
def _record_actions(obs, memory):
    """
    Read each action-history entry at most once.
    """
    hand = int(obs.hand_index)
    if memory.get("history_hand") != hand:
        memory["history_hand"] = hand
        memory["history_length"] = 0
    start = min(
        memory.get("history_length", 0),
        len(obs.action_history),
    )
    for entry in obs.action_history[start:]:
        if entry.is_me:
            continue
        memory["opp_actions"] += 1
        if entry.action == Action.RAISE_HALF:
            memory["opp_raises"] += 1
            memory["opp_half_raises"] += 1
        elif entry.action == Action.RAISE_POT:
            memory["opp_raises"] += 1
            memory["opp_pot_raises"] += 1
        elif entry.action == Action.CHECK_CALL:
            # amount == 0 represents a check.
            if getattr(entry, "amount", 0) == 0:
                memory["opp_checks"] += 1
            else:
                memory["opp_calls"] += 1
    memory["history_length"] = len(obs.action_history)
def _opponent_aggression(memory):
    actions = memory.get("opp_actions", 0)
    raises = memory.get("opp_raises", 0)
    # Bayesian smoothing prevents extreme conclusions after two hands.
    return (raises + 3.0) / (actions + 12.0)
def _observed_pressure_fold_rate(memory):
    pressured = memory.get("pressured_hands", 0)
    folds = memory.get("pressure_folds", 0)
    return (folds + 2.0) / (pressured + 6.0)
def _last_opponent_action(obs):
    for entry in reversed(obs.action_history):
        if not entry.is_me:
            return entry
    return None
def _opponent_just_checked(obs):
    entry = _last_opponent_action(obs)
    return (
        entry is not None
        and entry.action == Action.CHECK_CALL
        and getattr(entry, "amount", 0) == 0
    )
def _facing_raise(obs):
    entry = _last_opponent_action(obs)
    return (
        obs.to_call > 0
        and entry is not None
        and entry.action in (Action.RAISE_HALF, Action.RAISE_POT)
    )
def _facing_pot_raise(obs):
    entry = _last_opponent_action(obs)
    return (
        obs.to_call > 0
        and entry is not None
        and entry.action == Action.RAISE_POT
    )
def _preflop_equity(cards):
    """
    Fast preflop approximation for Short Deck 5-card PLO.
    Rewards pairs, high cards, connected cards, and suits with at least
    two cards. It is intended for decisions, not as an exact equity model.
    """
    ranks = [card >> 2 for card in cards]
    suits = [card & 3 for card in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    unique = sorted(rank_counts)
    strength = 0.34
    # High-card quality
    strength += 0.16 * (sum(ranks) / 40.0)
    multiplicities = sorted(rank_counts.values(), reverse=True)
    if multiplicities[0] >= 4:
        strength += 0.25
    elif multiplicities[0] == 3:
        strength += 0.16
    elif multiplicities[0] == 2:
        pair_count = sum(
            1 for count in multiplicities if count == 2
        )
        strength += 0.075 if pair_count == 1 else 0.135
    # Omaha flush potential requires at least two hole cards of a suit.
    useful_suits = sum(
        1 for count in suit_counts.values() if count >= 2
    )
    strength += min(0.09, useful_suits * 0.04)
    # Connectivity
    connection = 0.0
    for i, first in enumerate(unique):
        for second in unique[i + 1:]:
            gap = second - first
            if gap == 1:
                connection += 0.024
            elif gap == 2:
                connection += 0.011
    strength += min(connection, 0.12)
    # Ace and A-6-7-8-9 possibilities
    if 8 in rank_counts:
        strength += 0.025
        wheel_cards = sum(
            rank in rank_counts for rank in (0, 1, 2, 3)
        )
        strength += 0.007 * wheel_cards
    if max(ranks) <= 4 and connection < 0.04:
        strength -= 0.045
    return _clamp(strength, 0.25, 0.80)
def _postflop_equity(obs):
    """
    Monte Carlo showdown equity against a random legal opponent hand.
    On the river, only the opponent's five cards are sampled. Before the
    river, the missing board cards are also sampled.
    """
    board = list(obs.board)
    hole = list(obs.hole_cards)
    missing = 5 - len(board)
    known = set(hole)
    known.update(board)
    unseen = [
        card for card in range(36)
        if card not in known
    ]
    cards_needed = 5 + missing
    if len(unseen) < cards_needed:
        return 0.5
    trials = _MC_TRIALS.get(len(board), 12)
    # Spend a few more samples on large decisions.
    if obs.to_call >= 18 or obs.pot >= 55:
        trials += 5
    rng = _rng_for(obs, extra=43)
    hero_score = None
    if missing == 0:
        hero_score = omaha_score(hole, board)
    wins = 0.0
    for _ in range(trials):
        sampled = rng.sample(unseen, cards_needed)
        opponent_hole = sampled[:5]
        final_board = board + sampled[5:]
        my_score = (
            hero_score
            if hero_score is not None
            else omaha_score(hole, final_board)
        )
        opponent_score = omaha_score(
            opponent_hole,
            final_board,
        )
        if my_score > opponent_score:
            wins += 1.0
        elif my_score == opponent_score:
            wins += 0.5
    return wins / trials
def _equity(obs):
    if obs.street == Street.PREFLOP:
        equity = _preflop_equity(obs.hole_cards)
        if obs.is_button:
            equity += 0.015
        return _clamp(equity)
    return _postflop_equity(obs)
def _mark_pressure(obs, memory):
    hand = int(obs.hand_index)
    if memory.get("pressure_hand") != hand:
        memory["pressure_hand"] = hand
        memory["pressured_hands"] += 1
def _raise(obs, memory, action):
    if action in obs.legal_actions:
        _mark_pressure(obs, memory)
        return action
    return None
def _value_raise(obs, memory, equity, mix):
    """
    Pot raises represent strong hands most of the time.
    Some strong hands use half-pot so our sizing does not reveal the exact
    strength of our hand.
    """
    if equity >= 0.76:
        if mix < 0.78 and Action.RAISE_POT in obs.legal_actions:
            return _raise(obs, memory, Action.RAISE_POT)
        if Action.RAISE_HALF in obs.legal_actions:
            return _raise(obs, memory, Action.RAISE_HALF)
        if Action.RAISE_POT in obs.legal_actions:
            return _raise(obs, memory, Action.RAISE_POT)
    if equity >= 0.63:
        if Action.RAISE_HALF in obs.legal_actions:
            return _raise(obs, memory, Action.RAISE_HALF)
        if mix < 0.35 and Action.RAISE_POT in obs.legal_actions:
            return _raise(obs, memory, Action.RAISE_POT)
    return None
def _act(obs, memory):
    _record_actions(obs, memory)
    equity = _equity(obs)
    aggression = _opponent_aggression(memory)
    pressure_fold_rate = _observed_pressure_fold_rate(memory)
    rng = _rng_for(obs, extra=97)
    mix = rng.random()
    # Small noise prevents a perfectly fixed decision boundary.
    equity += (rng.random() - 0.5) * 0.035
    equity = _clamp(equity)
    facing_raise = _facing_raise(obs)
    facing_pot_raise = _facing_pot_raise(obs)
    checked_to_us = _opponent_just_checked(obs)
    # --------------------------------------------------------------
    # Nothing to call
    # --------------------------------------------------------------
    if obs.to_call == 0:
        # Trap some strong hands. The target should not be allowed to
        # interpret every check as a weak range.
        trap_frequency = 0.28 if aggression >= 0.24 else 0.16
        if equity >= 0.74 and mix < trap_frequency:
            return Action.CHECK_CALL
        value_action = _value_raise(
            obs,
            memory,
            equity,
            mix,
        )
        if value_action is not None:
            return value_action
        # Primary exploit: the attached agent makes equity/price-based
        # folds. Apply inexpensive half-pot pressure after it checks.
        if (
            checked_to_us
            and obs.street != Street.PREFLOP
            and Action.RAISE_HALF in obs.legal_actions
        ):
            if equity >= 0.46:
                return _raise(
                    obs,
                    memory,
                    Action.RAISE_HALF,
                )
            # Controlled bluffing. Increase only if actual results show
            # that the opponent is folding to our pressure.
            bluff_rate = 0.10
            if pressure_fold_rate >= 0.48:
                bluff_rate += min(
                    0.16,
                    (pressure_fold_rate - 0.48) * 0.65,
                )
            # River checks cap the target's range more meaningfully, but
            # do not bluff every river.
            if obs.street == Street.RIVER:
                bluff_rate += 0.045
            if (
                0.25 <= equity < 0.46
                and mix < bluff_rate
            ):
                    return _raise(
                    obs,
                    memory,
Action.RAISE_HALF,
                )
        # Preflop: do not attack this opponent indiscriminately. Its
        # preflop model appears designed to defend many hands.
        if (
            obs.street == Street.PREFLOP
            and obs.is_button
            and equity >= 0.57
            and Action.RAISE_HALF in obs.legal_actions
            and mix < 0.55
        ):
            return _raise(
                obs,
                memory,
                Action.RAISE_HALF,
            )
        return Action.CHECK_CALL
    # --------------------------------------------------------------
    # Facing a bet or raise
    # --------------------------------------------------------------
    pot_after_call = max(
        1.0,
        float(obs.pot + obs.to_call),
    )
    pot_odds = float(obs.to_call) / pot_after_call
    # The target's bluffing code is restrained, so its raises deserve
    # respect—especially pot-sized raises and late-street raises.
    required_margin = 0.025
    if facing_raise:
        required_margin += 0.055
    if facing_pot_raise:
        required_margin += 0.045
    if obs.street == Street.TURN and facing_raise:
        required_margin += 0.025
    if obs.street == Street.RIVER and facing_raise:
        required_margin += 0.055
    # If observed aggression becomes unusually high, some raises are more
    # likely to be bluffs or thin value.
    if aggression >= 0.36:
        required_margin -= 0.035
    # Do not fold too frequently to tiny bets. That would encourage the
    # attached bot's fold-rate-dependent bluffing.
    if obs.to_call <= obs.big_blind:
        required_margin -= 0.06
    adjusted_equity = equity
    # Raise strong hands, including occasional re-raises.
    if adjusted_equity >= 0.73:
        value_action = _value_raise(
            obs,
            memory,
            adjusted_equity,
            mix,
        )
        if value_action is not None:
            return value_action
    if adjusted_equity >= pot_odds + required_margin:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return _fallback(obs)
def choose_action(obs, memory):
    try:
        action = _act(obs, memory)
        if action in obs.legal_actions:
            return action
    except Exception:
        # An exception counts as an error, so never let one escape.
        pass
    return _fallback(obs)
def on_hand_end(result, memory):
    try:
        memory["hands"] = memory.get("hands", 0) + 1
        memory["net"] = (
            memory.get("net", 0) + int(result.my_net)
        )
        hand = int(result.hand_index)
        # If we raised, won, and did not reach showdown, the opponent
        # folded after pressure during this hand.
        if (
            memory.get("pressure_hand") == hand
            and result.winner == "me"
            and not result.went_to_showdown
        ):
            memory["pressure_folds"] += 1
        if memory.get("pressure_hand") == hand:
            memory["pressure_hand"] = -1
    except Exception:
        # on_hand_end exceptions also count as judge errors.
        return

# ===== END OF YOUR CODE =====

# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5004 to avoid conflicts with v.3 on port 5008
    port = int(os.environ.get("PORT", "5004"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
