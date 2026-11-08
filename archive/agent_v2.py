"""
agent.py - YOUR poker bot. This one file is a complete web server.

PORT = 5006!!!!

!!!!! REVERT TO agent_v1.py IF PERFORMANCE IS WORSE !!!!!

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
AGENT_NAME = "Gambling Addict v.2"

# ===== YOUR CODE HERE =====

# ----------------------------------------------------------------------------
# Strategy notes
# ----------------------------------------------------------------------------
#
# This bot plays Heads-Up Short Deck 5-Card Pot-Limit Omaha.
#
# Important rules:
# - Always return an action from obs.legal_actions.
# - CHECK_CALL means check when to_call is zero, otherwise it means call.
# - A final Omaha hand uses exactly 2 hole cards and exactly 3 board cards.
# - omaha_score() handles both the Omaha rule and Short Deck hand rankings,
#   but it can only be used once all five board cards are available.
# - The bot has a limited total time bank, so simulations are intentionally
#   small and bounded.
#
# The strategy:
# - Estimate showdown equity with a small Monte Carlo simulation.
# - Use pot odds to decide whether calling is profitable.
# - Bet and raise strong hands for value.
# - Occasionally make smaller semi-bluff bets with reasonable equity.
# - Track simple opponent aggression statistics during the current half.
# - Fall back safely to CHECK_CALL if anything unexpected happens.
# ----------------------------------------------------------------------------


# Much lower counts to stay under the 8-second per-half time bank.
SIMULATIONS_BY_BOARD_SIZE = {
    0: 0,   # Preflop: use the fast heuristic only
    3: 3,   # Flop
    4: 5,   # Turn
    5: 8,   # River
}


def new_half(config, memory):
    """
    Called once at the beginning of each half-match.

    A half contains 1,000 hands. Memory is reset between halves, so only
    store statistics useful for the current opponent during this half.
    """
    memory.clear()

    memory["hands"] = 0
    memory["net_chips"] = 0

    # Simple opponent-model statistics.
    memory["opp_actions"] = 0
    memory["opp_raises"] = 0

    # Gambling Addict v.2 - Additional tracking statistics for opponent model
    memory["opp_half_raises"] = 0
    memory["opp_pot_raises"] = 0
    memory["opp_calls_checks"] = 0
    memory["opp_folds"] = 0

    # Used to ensure we do not count the same history entry repeatedly.
    memory["last_history_hand"] = -1
    memory["last_history_length"] = 0


def on_hand_end(result, memory):
    """
    Called when a hand ends.

    This function is intentionally conservative. An exception in this
    function counts as a judge error, so bookkeeping must never crash the bot.
    """
    try:
        memory["hands"] = memory.get("hands", 0) + 1
        memory["net_chips"] = memory.get("net_chips", 0) + result.my_net
    except Exception:
        pass


def choose_action(obs, memory):
    """
    Return one legal action.

    The try/except is important because an invalid action or uncaught
    exception counts as an error. CHECK_CALL is always legal and is used as
    the safe fallback.
    """
    try:
        record_opponent_action(obs, memory)

        # ---------------------------------------------------------------
        # Fast paths: no equity simulation needed.
        # ---------------------------------------------------------------

        # If checking is free and we have already seen 4 raises, cannot raise.
        # Checking is the only sensible zero-cost action.
        if obs.to_call == 0 and obs.raises_this_street >= 4:
            return Action.CHECK_CALL

        # Preflop: use the quick heuristic rather than Monte Carlo.
        if len(obs.board) == 0:
            equity = preflop_strength(obs.hole_cards)
        else:
            equity = estimate_equity(obs)

        aggression = opponent_aggression(memory)

        if obs.to_call == 0:
            action = choose_when_checked_to(obs, equity, aggression)
        else:
            action = choose_facing_bet(obs, equity, aggression)

        if action in obs.legal_actions:
            return action

    except Exception:
        pass

    # CHECK_CALL is documented as always legal.
    return Action.CHECK_CALL


# ----------------------------------------------------------------------------
# Opponent modelling
# ----------------------------------------------------------------------------

# Gambling Addict v.2 - Now tracks ALL opponent actions that agent hasn't seen before
def record_opponent_action(obs, memory):
    """
    Process every previously unseen opponent action.

    Uses a history cursor to prevent double-counting across decisions
    and resets the cursor when a new hand begins.
    """
    try:
        history = obs.action_history
        hand_index = obs.hand_index
        history_length = len(history)

        if memory.get("last_history_hand") != hand_index:
            start_index = 0
        else:
            start_index = memory.get("last_history_length", 0)

            if start_index > history_length:
                start_index = 0

        for entry in history[start_index:]:
            if entry.is_me:
                continue

            memory["opp_actions"] = memory.get("opp_actions", 0) + 1

            if entry.action == Action.RAISE_HALF:
                memory["opp_raises"] = memory.get("opp_raises", 0) + 1
                memory["opp_half_raises"] = memory.get("opp_half_raises", 0) + 1

            elif entry.action == Action.RAISE_POT:
                memory["opp_raises"] = memory.get("opp_raises", 0) + 1
                memory["opp_pot_raises"] = memory.get("opp_pot_raises", 0) + 1

            elif entry.action == Action.CHECK_CALL:
                memory["opp_calls_checks"] = memory.get("opp_calls_checks", 0) + 1

            elif entry.action == Action.FOLD:
                memory["opp_folds"] = memory.get("opp_folds", 0) + 1

        memory["last_history_hand"] = hand_index
        memory["last_history_length"] = history_length

    except Exception:
        pass



def opponent_aggression(memory):
    """
    Return an estimated opponent raise frequency.

    Use a neutral value while little information is available. The value is
    capped so that a few unusual hands cannot cause an extreme adjustment.
    """
    actions = memory.get("opp_actions", 0)
    raises = memory.get("opp_raises", 0)

    if actions < 8:
        return 0.30

    raise_rate = raises / max(1, actions)

    return max(0.10, min(0.65, raise_rate))


# ----------------------------------------------------------------------------
# Equity estimation
# ----------------------------------------------------------------------------

def estimate_equity(obs):
    """
    Estimate showdown equity by sampling unknown opponent cards and missing
    board cards.

    Each simulation:
      1. Samples five legal opponent hole cards.
      2. Completes the board to five cards.
      3. Scores both hands with omaha_score.
      4. Counts wins as 1.0 and ties as 0.5.

    The evaluator applies the Short Deck rankings and mandatory Omaha 2+3
    card rule automatically.
    """
    """
    Fast equity estimate.

    Preflop uses only the inexpensive structural hand heuristic.
    Postflop uses a deliberately small Monte Carlo sample.
    """
    hero_hole = list(obs.hole_cards)
    visible_board = list(obs.board)

    # Preflop Monte Carlo is the most expensive and noisiest stage because
    # all five board cards are unknown. Use the fast heuristic instead.
    if len(visible_board) == 0:
        return preflop_strength(hero_hole)

    known_cards = set(hero_hole + visible_board)
    unseen_cards = [card for card in range(36) if card not in known_cards]

    missing_board_cards = 5 - len(visible_board)
    simulations = simulation_count(obs)

    rng = random.Random(observation_seed(obs))
    points = 0.0

    hero_river_score = None
    if len(visible_board) == 5:
        hero_river_score = omaha_score(hero_hole, visible_board)


    for _ in range(simulations):
        sampled_cards = rng.sample(
            unseen_cards,
            5 + missing_board_cards,
        )

        opponent_hole = sampled_cards[:5]
        full_board = visible_board + sampled_cards[5:]

        hero_score = (
            hero_river_score
            if hero_river_score is not None
            else omaha_score(hero_hole, full_board)
        )
        opponent_score = omaha_score(opponent_hole, full_board)

        if hero_score > opponent_score:
            points += 1.0
        elif hero_score == opponent_score:
            points += 0.5

    equity = points / simulations

    # Small positional adjustment postflop.
    if obs.is_button:
        equity += 0.01
    else:
        equity -= 0.005

    return max(0.02, min(0.98, equity))

def simulation_count(obs):
    """
    Keep computations cheap enough for roughly 2–4 ms per decision.
    """
    count = SIMULATIONS_BY_BOARD_SIZE.get(len(obs.board), 3)

    # Spend a few more simulations only for meaningful decisions.
    if obs.to_call >= 20:
        count += 2

    # Hard cap: do not let one difficult decision consume the time bank.
    return min(count, 10)


def observation_seed(obs):
    """
    Create a repeatable random seed from the current observation.

    Repeatable sampling makes the same game state produce the same equity
    estimate instead of changing because of unrelated random-number state.
    """
    seed = 2166136261

    values = (
        list(obs.hole_cards)
        + list(obs.board)
        + [
            int(obs.hand_index),
        ]
    )

    for value in values:
        seed ^= int(value) + 1
        seed = (seed * 16777619) & 0xFFFFFFFF

    return seed


# ----------------------------------------------------------------------------
# Preflop hand evaluation
# ----------------------------------------------------------------------------

def preflop_strength(hole_cards):
    """
    Produce a rough preflop equity-like estimate.

    This is not a hand evaluator. It rewards characteristics that tend to be
    useful in Short Deck Omaha:
    - High cards, especially aces, kings, and queens.
    - Pairs, particularly high pairs.
    - Suited hole-card combinations.
    - Connected ranks that can participate in straights.
    """
    ranks = [card >> 2 for card in hole_cards]
    suits = [card & 3 for card in hole_cards]

    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Begin around an average, playable baseline.
    strength = 0.27

    # Rank values are:
    # 0=6, 1=7, 2=8, 3=9, 4=T, 5=J, 6=Q, 7=K, 8=A.
    high_cards = sum(rank >= 6 for rank in ranks)
    strength += 0.030 * high_cards

    strength += 0.020 * rank_counts.get(8, 0)  # Aces
    strength += 0.010 * rank_counts.get(7, 0)  # Kings

    # Reward pairs, with extra value for premium pairs.
    for rank, count in rank_counts.items():
        if count == 2:
            strength += 0.070

            if rank == 8:
                strength += 0.100
            elif rank == 7:
                strength += 0.060
            elif rank == 6:
                strength += 0.035

        elif count >= 3:
            # Trips in the hole are useful but have diminished additional
            # value because they block cards needed to improve further.
            strength += 0.045

    # Count same-suit pairs among our five hole cards.
    suited_combinations = 0

    for count in suit_counts.values():
        suited_combinations += count * (count - 1) // 2

    strength += min(0.09, suited_combinations * 0.018)

    # Reward connected and near-connected rank combinations.
    unique_ranks = sorted(rank_counts.keys())
    connectivity = 0.0

    for index, low_rank in enumerate(unique_ranks):
        for high_rank in unique_ranks[index + 1:]:
            gap = high_rank - low_rank

            if gap == 1:
                connectivity += 0.016
            elif gap == 2:
                connectivity += 0.007

    # Short Deck permits the ace-low straight A-6-7-8-9.
    if 8 in rank_counts:
        low_cards_present = sum(
            rank in rank_counts for rank in (0, 1, 2, 3)
        )
        connectivity += 0.008 * low_cards_present

    strength += min(0.10, connectivity)

    return max(0.20, min(0.84, strength))


# ----------------------------------------------------------------------------
# Betting decisions
# ----------------------------------------------------------------------------

def choose_when_checked_to(obs, equity, aggression):
    """
    Decide whether to check, half-pot bet, or pot-sized bet when checking is
    free.
    """
    # Very strong hands: build the pot and extract value.
    if equity >= 0.76 and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT

    # Good hands: bet smaller for value and protection.
    if equity >= 0.61 and Action.RAISE_HALF in obs.legal_actions:
        return Action.RAISE_HALF

    # Do not bluff frequently on the river because no card remains to improve.
    if len(obs.board) == 5:
        return Action.CHECK_CALL

    # Semi-bluff with hands that retain reasonable showdown equity. Bluff less
    # often against opponents who have been raising frequently.
    semi_bluff_threshold = 0.48 + 0.08 * aggression

    if (
        equity >= semi_bluff_threshold
        and Action.RAISE_HALF in obs.legal_actions
    ):
        return Action.RAISE_HALF

    return Action.CHECK_CALL


def choose_facing_bet(obs, equity, aggression):
    """
    Decide whether to fold, call, raise half-pot, or raise pot when facing a
    wager.

    Pot odds are the minimum equity required for an immediate break-even call:
        to_call / (pot + to_call)
    """
    final_pot_if_called = obs.pot + obs.to_call
    pot_odds = obs.to_call / max(1, final_pot_if_called)

    # Earlier streets have more hidden information and future betting risk.
    uncertainty = uncertainty_margin(len(obs.board))

    # Aggressive opponents may bluff more often, so call slightly wider.
    aggression_discount = max(0.0, aggression - 0.30) * 0.10

    call_threshold = pot_odds + uncertainty - aggression_discount
    call_threshold = max(0.05, min(0.92, call_threshold))

    # Raise very strong holdings for value.
    if equity >= 0.82 and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT

    # Use a smaller value raise with solid equity when the current bet is not
    # already extremely large.
    if (
        equity >= 0.68
        and obs.to_call <= max(8, obs.pot // 2)
        and Action.RAISE_HALF in obs.legal_actions
    ):
        return Action.RAISE_HALF

    # Standard pot-odds call.
    if equity >= call_threshold:
        return Action.CHECK_CALL

    # Defend very cheap calls slightly wider.
    if obs.to_call <= obs.big_blind and equity >= 0.24:
        return Action.CHECK_CALL

    if Action.FOLD in obs.legal_actions:
        return Action.FOLD

    return Action.CHECK_CALL


def uncertainty_margin(board_size):
    """
    Return an additional equity buffer for earlier streets.

    On the river all cards are known, so the equity estimate is much more
    directly comparable to the immediate pot odds.
    """
    if board_size == 0:
        return 0.065

    if board_size == 3:
        return 0.040

    if board_size == 4:
        return 0.020

    return 0.005


# ===== END OF YOUR CODE =====

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
    port = int(os.environ.get("PORT", "5006"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
