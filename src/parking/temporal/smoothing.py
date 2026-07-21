"""Per-spot hysteresis to suppress single-frame flicker (plan §5).

Never flip a spot's state on one frame — a pedestrian, a car mid-park, or a
moving shadow all cause transient misclassifications. A spot only changes state
after ``k`` consecutive agreeing frames.

Tune ``k`` against frame rate: at 30 fps, ``k=5`` is ~0.17 s of confirmation —
enough to reject transients, fast enough to feel live.
"""

from __future__ import annotations

from collections.abc import Sequence

from parking.types import Occupancy, Prediction, SpotResult


class SpotState:
    """Hysteresis state machine for a single spot.

    Directly follows the plan's reference implementation: require ``k`` frames
    of agreement on a *new* prediction before committing to it.
    """

    def __init__(self, spot_id: str, k: int = 5, initial: Occupancy = Occupancy.EMPTY):
        self.spot_id = spot_id
        self.state = initial
        self.k = k
        self.streak = 0
        self.pending: Occupancy | None = None

    def update(self, pred: Occupancy) -> Occupancy:
        if pred == self.state:
            self.streak, self.pending = 0, None
            return self.state
        if pred == self.pending:
            self.streak += 1
        else:
            self.pending, self.streak = pred, 1
        if self.streak >= self.k:
            self.state, self.streak, self.pending = pred, 0, None
        return self.state


class SmoothingBank:
    """Holds one :class:`SpotState` per spot and smooths a whole frame's worth
    of predictions at once, preserving spot order."""

    def __init__(self, spot_ids: Sequence[str], k: int = 5):
        self.states = {sid: SpotState(sid, k=k) for sid in spot_ids}

    def update(
        self, spot_ids: Sequence[str], preds: Sequence[Prediction]
    ) -> list[SpotResult]:
        results: list[SpotResult] = []
        for sid, pred in zip(spot_ids, preds):
            st = self.states.get(sid)
            if st is None:  # spot appeared at runtime (shouldn't normally happen)
                st = self.states[sid] = SpotState(sid, k=next(iter(self.states.values())).k)
            smoothed = st.update(pred.label)
            results.append(SpotResult(spot_id=sid, state=smoothed, raw=pred))
        return results
