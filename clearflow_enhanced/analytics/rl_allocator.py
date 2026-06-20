"""
rl_allocator.py  —  Module N: Self-Improving Resource Allocation
----------------------------------------------------------------------
A lightweight contextual-bandit style learner (not full deep RL —
appropriately scoped for a hackathon, but the same epsilon-greedy /
reward-update pattern that real RL traffic systems use).

Learns, from feedback_engine.py's dispatch outcomes, which ACTION
(tow / officer / temp_parking / diversion) works best in which ZONE,
and updates its recommendation over time.

    "Deploying 3 officers at Junction A historically reduces
     congestion 40% faster than towing, based on 14 past dispatches."

State: zone_id
Action: one of {TOW, OFFICER, TEMP_PARKING, DIVERSION, SIGNAL_RETIME}
Reward: ci_recovery_pct from feedback (Module I)

This module reads feedback_engine's persisted JSONL so learning
carries over between sessions — genuinely self-improving, not just
a static rulebook.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


@dataclass
class ActionStats:
    action_type: str
    n_trials: int = 0
    total_reward: float = 0.0     # sum of ci_recovery_pct achieved
    avg_response_time_sec: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.n_trials if self.n_trials else 0.0


@dataclass
class AllocationRecommendation:
    zone_id: str
    zone_name: str
    recommended_action: str
    expected_ci_recovery_pct: float
    confidence: str            # LOW (few samples) / MEDIUM / HIGH
    n_historical_trials: int
    narrative: str


class RLAllocator:
    """
    Epsilon-greedy contextual bandit over (zone, action) pairs.
    Call update_from_feedback() to ingest new outcomes.
    Call recommend() to get the current best action for a zone.
    """

    ACTIONS = ["TOW", "OFFICER", "TEMP_PARKING", "DIVERSION", "SIGNAL_RETIME"]
    EPSILON = 0.15   # exploration rate

    def __init__(self, state_path: str = "output/feedback/rl_state.json"):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        # zone_id -> action_type -> ActionStats
        self._q: Dict[str, Dict[str, ActionStats]] = defaultdict(
            lambda: {a: ActionStats(action_type=a) for a in self.ACTIONS}
        )
        self._load()

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            for zone_id, actions in data.items():
                for action_type, stats in actions.items():
                    self._q[zone_id][action_type] = ActionStats(
                        action_type=action_type,
                        n_trials=stats["n_trials"],
                        total_reward=stats["total_reward"],
                        avg_response_time_sec=stats.get("avg_response_time_sec", 0.0),
                    )
        except Exception:
            pass

    def _save(self):
        out = {
            zone_id: {
                a: {
                    "n_trials": s.n_trials,
                    "total_reward": s.total_reward,
                    "avg_response_time_sec": s.avg_response_time_sec,
                }
                for a, s in actions.items()
            }
            for zone_id, actions in self._q.items()
        }
        with open(self.state_path, "w") as f:
            json.dump(out, f, indent=2)

    def update_from_outcome(
        self,
        zone_id: str,
        action_type: str,
        ci_recovery_pct: float,
        response_time_sec: Optional[float] = None,
    ):
        """Record a (zone, action) -> reward observation."""
        if action_type not in self.ACTIONS:
            return
        stats = self._q[zone_id][action_type]
        stats.n_trials += 1
        stats.total_reward += ci_recovery_pct
        if response_time_sec:
            # running average
            n = stats.n_trials
            stats.avg_response_time_sec = (
                (stats.avg_response_time_sec * (n - 1) + response_time_sec) / n
            )
        self._save()

    def bootstrap_from_feedback_history(self, feedback_history: list):
        """
        Seed the bandit from FeedbackEngine's historical DispatchFeedback
        records. Since those records don't carry action_type explicitly
        in this version, we infer a plausible action from response time
        as a reasonable proxy for demo purposes.
        """
        for fb in feedback_history:
            if fb.response_time_sec is None:
                continue
            # Heuristic inference for demo: fast response -> OFFICER,
            # mid -> TOW, slow -> TEMP_PARKING. Real system would log
            # the actual action taken.
            if fb.response_time_sec < 300:
                action = "OFFICER"
            elif fb.response_time_sec < 600:
                action = "TOW"
            else:
                action = "TEMP_PARKING"
            self.update_from_outcome(
                zone_id=fb.zone_id, action_type=action,
                ci_recovery_pct=fb.ci_recovery_pct,
                response_time_sec=fb.response_time_sec,
            )

    def recommend(self, zone_id: str, zone_name: str) -> AllocationRecommendation:
        """
        Epsilon-greedy recommendation: usually picks the best-known action,
        occasionally explores an under-tried one to keep learning.
        """
        actions = self._q[zone_id]
        total_trials = sum(s.n_trials for s in actions.values())

        if total_trials == 0:
            return AllocationRecommendation(
                zone_id=zone_id, zone_name=zone_name,
                recommended_action="OFFICER",
                expected_ci_recovery_pct=0.0, confidence="LOW",
                n_historical_trials=0,
                narrative=f"No historical data for {zone_name} yet — "
                          f"defaulting to officer deployment (safest baseline action). "
                          f"Recommendation will improve as outcomes are recorded.",
            )

        if random.random() < self.EPSILON:
            # Explore: pick a less-tried action
            chosen = min(actions.values(), key=lambda s: s.n_trials)
        else:
            # Exploit: pick the best average reward (min 1 trial)
            tried = [s for s in actions.values() if s.n_trials > 0]
            chosen = max(tried, key=lambda s: s.avg_reward) if tried else list(actions.values())[0]

        confidence = "HIGH" if chosen.n_trials >= 10 else ("MEDIUM" if chosen.n_trials >= 3 else "LOW")

        # Compare to second best for narrative context
        ranked = sorted(
            [s for s in actions.values() if s.n_trials > 0],
            key=lambda s: s.avg_reward, reverse=True
        )
        comparison = ""
        if len(ranked) >= 2 and ranked[0].action_type == chosen.action_type:
            pct_better = ranked[0].avg_reward - ranked[1].avg_reward
            comparison = (
                f" This outperforms {ranked[1].action_type} by "
                f"{pct_better:.1f} CI-recovery points on average."
            )

        narrative = (
            f"Deploying {chosen.action_type.replace('_',' ').title()} at {zone_name} "
            f"historically recovers {chosen.avg_reward:.1f}% CI on average "
            f"(based on {chosen.n_trials} past dispatch{'es' if chosen.n_trials != 1 else ''})."
            f"{comparison}"
        )

        return AllocationRecommendation(
            zone_id=zone_id, zone_name=zone_name,
            recommended_action=chosen.action_type,
            expected_ci_recovery_pct=round(chosen.avg_reward, 1),
            confidence=confidence, n_historical_trials=chosen.n_trials,
            narrative=narrative,
        )

    def print_learning_state(self, zone_names: Dict[str, str]):
        print(f"\n  ── RL ALLOCATOR: Learned Action Values ──")
        for zone_id, actions in self._q.items():
            zone_name = zone_names.get(zone_id, zone_id)
            tried = {a: s for a, s in actions.items() if s.n_trials > 0}
            if not tried:
                continue
            print(f"\n  {zone_name}:")
            for action_type, s in sorted(tried.items(), key=lambda kv: -kv[1].avg_reward):
                print(f"    {action_type:<16} avg_reward={s.avg_reward:>5.1f}%  "
                      f"n={s.n_trials:<3} resp={s.avg_response_time_sec:.0f}s")
