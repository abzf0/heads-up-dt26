"""
agent.py - YOUR poker bot. This one file is a complete web server.

PORT = 5008

Strategy: calibrated preflop equity, opponent-sensitive value bets and calls.

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
AGENT_NAME = "Gambling Addict v.4"

# ===== YOUR CODE HERE =====

# Offline ridge fit against random five-card hands. Training data used the
# tournament's evaluator and Omaha 2+3/short-deck rules, not Texas Hold'em.
# 1,800 independent hands, 65 legal opponent/board samples per hand.
_PREFLOP_COEFFICIENTS = (
    0.3268757, -0.0796757, -0.0641181, -0.0308902, -0.0090759,
    0.0570487, 0.0378739, 0.0291623, 0.0259876, 0.0336873,
    0.0623633, -0.047782, -0.045887, 0.0, 0.0388157,
    0.0341723, 0.0261011, -0.0220811, -0.013634, 0.0107049,
    0.0050258, 0.0298179, 0.0200865, 0.0078385, 0.0128876,
    -0.0097763, -0.0149428, -0.0083922, 0.0202659, 0.0286581,
)

# Conservative bounded samples, emphasizing the river (where full board is
# known and all betting risk has been resolved). Adapted for the 8-second bank.
_TRIALS = {3: 18, 4: 24, 5: 38}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def preflop_strength(cards):
    ranks = [c >> 2 for c in cards]
    suits = [c & 3 for c in cards]
    counts = Counter(ranks)
    suit_counts = Counter(suits)
    unique = sorted(counts)

    f = [1.0]
    f.extend(counts.get(r, 0) / 3 for r in range(9))
    f.extend(sum(1 for n in counts.values() if n == v) for v in (2, 3, 4, 5))
    f.extend((sum((r / 8) ** 2 for r, n in counts.items() if n >= 2), len(unique) / 5))
    f.extend(sum(1 for n in suit_counts.values() if n >= v) for v in (2, 3, 4))
    f.append(sum(1 for c in cards if c >> 2 == 8 and suit_counts[c & 3] >= 2))
    f.append(sum(1 for c in cards if c >> 2 >= 6 and suit_counts[c & 3] >= 2))
    for d in (1, 2, 3, 4):
        f.append(sum(1 for a in unique for b in unique if b - a == d) / 4)
    f.append(sum(1 for r in (0, 1, 2, 3) if r in counts and 8 in counts) / 4)
    f.append(max(sum(r in counts for r in range(a, a + 5)) for a in range(5)) / 5)
    f.extend((unique[0] / 8, unique[-1] / 8, (unique[-1] - unique[0]) / 8))
    return _clamp(sum(a * b for a, b in zip(f, _PREFLOP_COEFFICIENTS)), 0.17, 0.78)


def new_half(config, memory):
    memory.clear()
    memory['opp_by_street'] = [[0, 0] for _ in range(4)]
    memory['last_hand'] = -1
    memory['last_history_len'] = 0
    memory['opp_raised_this_hand'] = [0, 0, 0, 0]
    memory['response_pending'] = False
    memory['raise_responses'] = 0
    memory['raise_folds'] = 0
    memory['showdowns'] = 0
    memory['hands'] = 0
    memory['last_my_hole'] = ()


def on_hand_end(result, memory):
    memory['hands'] = memory.get('hands', 0) + 1
    if result.went_to_showdown:
        memory['showdowns'] = memory.get('showdowns', 0) + 1
    # If our last bet was unanswered, the opponent necessarily folded.
    if memory.get('response_pending') and result.winner == 'me' and not result.went_to_showdown:
        memory['raise_responses'] += 1
        memory['raise_folds'] += 1
    memory['response_pending'] = False


def _track(obs, memory):
    if memory.get('last_hand') != obs.hand_index:
        memory['last_hand'] = obs.hand_index
        memory['last_history_len'] = 0
        memory['opp_raised_this_hand'] = [0, 0, 0, 0]
        memory['response_pending'] = False
    history = obs.action_history
    start = min(len(history), memory.get('last_history_len', 0))
    for act in history[start:]:
        if act.is_me:
            continue
        s = int(act.street)
        if act.action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory['opp_by_street'][s][0] += 1
            memory['opp_raised_this_hand'][s] += 1
        elif act.action == Action.CHECK_CALL:
            memory['opp_by_street'][s][1] += 1
        if memory.get('response_pending'):
            memory['response_pending'] = False
            memory['raise_responses'] += 1
    memory['last_history_len'] = len(history)


def _aggression(memory, street):
    r, c = memory['opp_by_street'][street]
    # Prior equivalent to 12 observations to avoid wild early adaptation.
    return (r + 3.4) / (r + c + 12.0)


def _fold_to_raise(memory):
    return (memory['raise_folds'] + 2.5) / (memory['raise_responses'] + 9.0)


def _seed(obs, extra=0):
    seed = 2166136261
    for v in (*obs.hole_cards, *obs.board, obs.hand_index, extra):
        seed = ((seed ^ (v + 13)) * 16777619) & 0xFFFFFFFF
    return seed


def _equity(obs, memory):
    if obs.street == Street.PREFLOP:
        return preflop_strength(obs.hole_cards)

    board = list(obs.board)
    known = set((*obs.hole_cards, *board))
    unseen = [c for c in range(36) if c not in known]
    missing = 5 - len(board)
    n = _TRIALS[len(board)]
    # Every independent call doesn't need to be more expensive; high-stakes
    # decisions and late-street decisions receive a little extra precision.
    if obs.to_call >= 15 or obs.pot >= 48:
        n += 7
    rng = random.Random(_seed(obs))
    hero_score = omaha_score(obs.hole_cards, board) if missing == 0 else None

    # A preflop raise narrows a tight opponent's range, but not a loose one's.
    preflop_raises = memory['opp_raised_this_hand'][0]
    preflop_tightness = _clamp((0.57 - _aggression(memory, 0)) * 2.6, 0.15, 1.0)
    # A river raise is usually much more informative than a flop bet.
    street_raises = memory['opp_raised_this_hand'][int(obs.street)]
    aggressive = _aggression(memory, int(obs.street))
    postflop_bias = (0.9 if len(board) == 5 else 0.35) * _clamp((0.65 - aggressive) * 1.9, 0.2, 1.0)

    wins = 0.0
    total_w = 0.0
    for _ in range(n):
        sample = rng.sample(unseen, 5 + missing)
        opp = sample[:5]
        final_board = board if not missing else board + sample[5:]
        mine = hero_score if hero_score is not None else omaha_score(obs.hole_cards, final_board)
        theirs = omaha_score(opp, final_board)

        w = 1.0
        if preflop_raises:
            # Posterior approximation: opponents who raise tend to have
            # above-average preflop starting hands, except habitual raisers.
            w *= _clamp(1.0 + preflop_tightness * (preflop_strength(opp) - 0.49) * 3.0,
                        0.42, 1.7)
        if street_raises:
            category = theirs >> 20
            strength = (0.025, 0.025, 0.14, 0.39, 0.60, 0.81, 0.96, 0.99, 1.0)[category]
            w *= _clamp(1.0 + postflop_bias * (strength - 0.50) * 1.8, 0.45, 1.55)
        wins += w * (1.0 if mine > theirs else 0.5 if mine == theirs else 0.0)
        total_w += w
    return _clamp(wins / total_w, 0.015, 0.985)


def _raise(obs, chosen, memory):
    if chosen in obs.legal_actions:
        if chosen in (Action.RAISE_POT, Action.RAISE_HALF):
            memory['response_pending'] = True
        return chosen
    return Action.CHECK_CALL


def _act(obs, memory):
    _track(obs, memory)
    if obs.to_call == 0 and Action.RAISE_HALF not in obs.legal_actions:
        return Action.CHECK_CALL

    equity = _equity(obs, memory)
    street = int(obs.street)
    aggr = _aggression(memory, street)
    foldrate = _fold_to_raise(memory)
    random_bluff = (_seed(obs, 93) % 1000) / 1000
    legal = obs.legal_actions

    # Most short-deck hands run close in raw equity, so only enlarge preflop
    # pots vs predictable passive ranges; do not do this against maniacs.
    pr, pc = memory['opp_by_street'][0]
    opp_preflop_raise_rate = _aggression(memory, 0)
    enough = _clamp((pr + pc) / 35.0, 0, 1)
    exploitable = enough * _clamp((0.36 - opp_preflop_raise_rate) / 0.19, 0, 1)
    # Preflop: calibrated showdown equity (~0.50 average). Most hands defend
    # the cheap blind but large raises require a materially stronger hand.
    if street == 0:
        if obs.to_call:
            pot_odds = obs.to_call / max(1, obs.pot + obs.to_call)
            if (equity < max(0.28, pot_odds + 0.035) and Action.FOLD in legal):
                return Action.FOLD
            if equity >= (0.68 - 0.05 * exploitable if obs.to_call >= 8 else 0.65 - 0.04 * exploitable) and Action.RAISE_POT in legal:
                return _raise(obs, Action.RAISE_POT, memory)
            if equity >= 0.615 - 0.05 * exploitable and obs.to_call <= 6 and Action.RAISE_HALF in legal:
                return _raise(obs, Action.RAISE_HALF, memory)
        else:
            if equity >= 0.625 - 0.04 * exploitable and Action.RAISE_POT in legal:
                return _raise(obs, Action.RAISE_POT, memory)
            if equity >= 0.585 - 0.065 * exploitable and Action.RAISE_HALF in legal:
                return _raise(obs, Action.RAISE_HALF, memory)
        return Action.CHECK_CALL

    pot_odds = obs.to_call / max(1, obs.pot + obs.to_call)
    # Opponents who rarely raise represent a stronger range when they do.
    tight_raises = max(0.0, 0.30 - aggr)
    if obs.to_call:
        # Protect against committing a stack on a thin, uncertain draw; the
        # river has no future streets, so calls only need to beat pot odds.
        margin = (0.065, 0.065, 0.05, 0.028)[street] + 0.12 * tight_raises
        margin += ((0.0, 0.045, 0.04, 0.032)[street] + 0.13 * tight_raises) * _clamp((0.62 - aggr) / 0.20, 0.0, 1.0)
        if street < 3 and obs.to_call >= 20:
            margin += 0.014
        # Some strong made hands should raise rather than just call.
        value_pot = 0.82 if street == 3 else 0.78
        value_half = 0.73 if street == 3 else 0.69
        if equity >= value_pot and Action.RAISE_POT in legal:
            return _raise(obs, Action.RAISE_POT, memory)
        if equity >= value_half and Action.RAISE_HALF in legal:
            return _raise(obs, Action.RAISE_HALF, memory)
        if equity + 0.002 >= pot_odds + margin:
            return Action.CHECK_CALL
        if Action.FOLD in legal:
            return Action.FOLD
        return Action.CHECK_CALL

    # When checked to, we can win by showdown or cause a fold. Bigger bets
    # extract value, smaller bets deny equity and conserve chips.
    if equity >= (0.68 if street == 3 else 0.67) and Action.RAISE_POT in legal:
        return _raise(obs, Action.RAISE_POT, memory)
    if equity >= (0.55 if street == 3 else 0.535) and Action.RAISE_HALF in legal:
        return _raise(obs, Action.RAISE_HALF, memory)
    if (equity >= 0.53 and foldrate > 0.40 and aggr < 0.43
            and Action.RAISE_HALF in legal):
        return _raise(obs, Action.RAISE_HALF, memory)
    # Avoid transparent never-bluff behavior, but only bluff when the measured
    # opponent is willing to fold. Hashing makes our mix reproducible.
    if (0.19 <= equity < 0.48 and foldrate > 0.43
            and random_bluff < 0.055 + 0.20 * (foldrate - 0.36)
            and Action.RAISE_HALF in legal and obs.pot <= 45):
        return _raise(obs, Action.RAISE_HALF, memory)
    return Action.CHECK_CALL


def choose_action(obs, memory):
    try:
        action = _act(obs, memory)
        if action in obs.legal_actions:
            return action
    except Exception:
        pass
    return Action.CHECK_CALL

# Compatibility with the optional v.1/v.2 local regression checks. These
# functions are not used by the tournament strategy.
def estimate_equity(obs):
    m = {}
    new_half(None, m)
    return _equity(obs, m)


def simulation_count(obs):
    return min(10, _TRIALS.get(len(obs.board), 0))


def opponent_aggression(memory):
    count = memory.get('opp_actions', 0)
    if count < 8:
        return 0.30
    return _clamp(memory.get('opp_raises', 0) / max(1, count), 0.10, 0.65)


def record_opponent_action(obs, memory):
    if memory.get('legacy_hand') != obs.hand_index:
        memory['legacy_hand'] = obs.hand_index
        memory['legacy_len'] = 0
    start = min(memory.get('legacy_len', 0), len(obs.action_history))
    for act in obs.action_history[start:]:
        if act.is_me:
            continue
        memory['opp_actions'] = memory.get('opp_actions', 0) + 1
        if act.action == Action.RAISE_HALF:
            memory['opp_raises'] = memory.get('opp_raises', 0) + 1
            memory['opp_half_raises'] = memory.get('opp_half_raises', 0) + 1
        elif act.action == Action.RAISE_POT:
            memory['opp_raises'] = memory.get('opp_raises', 0) + 1
            memory['opp_pot_raises'] = memory.get('opp_pot_raises', 0) + 1
        elif act.action == Action.CHECK_CALL:
            memory['opp_calls_checks'] = memory.get('opp_calls_checks', 0) + 1
        elif act.action == Action.FOLD:
            memory['opp_folds'] = memory.get('opp_folds', 0) + 1
    memory['legacy_len'] = len(obs.action_history)


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
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
