"""Per-bend state decode and stage-model selection (E3 step 4; plan §3.2, ENGINE_IO §4.2).

**Evidence.** For every bend the measured sweep (the dihedral between the
fitted parent and child OML planes; the Hough angle when the child slab could
only be seen near the hinge) is compared with the compiled nominal sweep and
with flat. ``formed_fraction = measured sweep / nominal sweep``. The dihedral is
signed by the side the child lies on (negative: it turned the other way about
the hinge axis); a negative sweep beyond the flat limits is a bend formed the
wrong way, FORMED or PARTIALLY_FORMED by its magnitude, whose angle then reads
above 180 and FAILs (:mod:`.metrology`). Otherwise:

- ``<= flat_fraction_max`` (or within ``flat_sweep_max_deg`` of flat): NOT_YET_FORMED;
- ``>= formed_fraction_min``: FORMED;
- in between: PARTIALLY_FORMED (unless its angle already passes the guard band,
  which makes it FORMED by definition: the contract forbids a passing angle on
  a partial bend);
- child flange not found near its hinge: UNKNOWN (OCCLUDED / LOW_DENSITY), and
  every bend below it in the tree UNKNOWN (BEND_STATE_UNKNOWN).

**Order prior** (soft). The tagged order with its מהלך groups gives each bend a
rank (all bends of one group share the group's first rank). A formed bend that
ranks above an unformed one was not expected by the order: it is reported
UNEXPECTED_FORMED (the stage decision then flags OUT_OF_ORDER_SUSPECTED). The
prior also breaks ties inside ``prior_band`` of a threshold, and is reported
per bend as ``prior_formed_probability``; it never overrides clear evidence.

**Stage model.** An order-consistent formed set that is a sequence prefix
decodes to ``S<k>``. Any other formed set uses a compiled stage with exactly
that formed set (``S<k>-<tag>``) when E1 provided one, else no compiled model
(``stage_model_id`` null: no heatmap; nominal lines come from the formed-set
transforms of :func:`domains.production.engine.compile.stages_from_compiled`).
With UNKNOWN bends the smallest compiled stage consistent with every known
state is used for display only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from domains.production.engine.inspect.config import InspectConfig
from domains.production.engine.inspect.part import PartView

__all__ = ["BendDecode", "decode_states", "choose_stage_model"]


@dataclass
class BendDecode:
    state: str
    reasons: tuple[str, ...]
    measured_sweep_deg: float | None
    formed_fraction: float | None
    prior: float | None


def _raw_state(sweep_meas: float | None, sweep_nom: float, cfg: InspectConfig) -> str | None:
    if sweep_meas is None:
        return None
    if sweep_meas < 0.0:
        # the child lies on the other side of its parent's plane: a kink within the flat limits is a flat
        # sheet; beyond them the bend was formed the wrong way, FORMED or PARTIALLY_FORMED by its magnitude
        # (its angle then reads above 180 and FAILs: :mod:`.metrology`)
        mag = -sweep_meas
        if mag <= cfg.flat_sweep_max_deg or mag / sweep_nom <= cfg.flat_fraction_max:
            return "NOT_YET_FORMED"
        return "FORMED" if mag / sweep_nom >= cfg.formed_fraction_min else "PARTIALLY_FORMED"
    frac = sweep_meas / sweep_nom
    if abs(sweep_meas) <= cfg.flat_sweep_max_deg or frac <= cfg.flat_fraction_max:
        return "NOT_YET_FORMED"
    if frac >= cfg.formed_fraction_min:
        return "FORMED"
    return "PARTIALLY_FORMED"


def decode_states(part: PartView, evidence: list[tuple[float | None, str | None]], cfg: InspectConfig
                  ) -> list[BendDecode]:
    """``evidence[j] = (measured sweep deg or None, UNKNOWN reason or None)`` per bend (sequence order)."""
    raw: list[str | None] = []
    for j, (sw, _) in enumerate(evidence):
        raw.append(_raw_state(sw, math.degrees(part.bends[j].sweep), cfg))
    ranks = [b.rank for b in part.bends]
    # order prior: share of lower-rank bends that are formed-like among the decided ones
    out: list[BendDecode] = []
    for j, (sw, reason) in enumerate(evidence):
        lower = [raw[i] for i in range(part.n) if ranks[i] < ranks[j] and raw[i] is not None]
        if lower:
            prior = sum(1 for s in lower if s in ("FORMED", "PARTIALLY_FORMED")) / len(lower)
        else:
            prior = 1.0
        state = raw[j]
        nom = math.degrees(part.bends[j].sweep)
        frac = None if sw is None else sw / nom
        if state is not None and frac is not None:
            # soft prior inside the tie band of a threshold
            if abs(frac - cfg.flat_fraction_max) <= cfg.prior_band and abs(sw) > cfg.flat_sweep_max_deg:
                state = "PARTIALLY_FORMED" if prior >= 0.5 else "NOT_YET_FORMED"
            elif abs(frac - cfg.formed_fraction_min) <= cfg.prior_band:
                state = "FORMED" if prior >= 0.5 else "PARTIALLY_FORMED"
        if state is None:
            out.append(BendDecode("UNKNOWN", (reason or "OCCLUDED",), sw, frac, prior))
        else:
            out.append(BendDecode(state, (), sw, frac, prior))
    # unexpected: formed while a lower-ranked bend is not yet formed
    unformed_ranks = [ranks[i] for i, d in enumerate(out) if d.state == "NOT_YET_FORMED"]
    if unformed_ranks:
        lowest = min(unformed_ranks)
        for i, d in enumerate(out):
            if d.state == "FORMED" and ranks[i] > lowest:
                out[i] = BendDecode("UNEXPECTED_FORMED", d.reasons, d.measured_sweep_deg, d.formed_fraction, d.prior)
    return out


def choose_stage_model(part: PartView, states: list[str]) -> tuple[str | None, bool]:
    """(stage_model_id, is_exact) for the decoded states (module docstring).

    ``is_exact`` is False when the id was chosen for display only (UNKNOWN bends).
    """
    formed = [part.bends[i].uid for i, s in enumerate(states) if s in ("FORMED", "PARTIALLY_FORMED", "UNEXPECTED_FORMED")]
    unformed = {part.bends[i].uid for i, s in enumerate(states) if s == "NOT_YET_FORMED"}
    unknown = [i for i, s in enumerate(states) if s == "UNKNOWN"]
    if not unknown:
        k = len(formed)
        if formed == [b.uid for b in part.bends[:k]] and "UNEXPECTED_FORMED" not in states:
            return f"S{k}", True
        match = part.stage_for_formed(formed)
        if match is not None:
            return match.stage_model_id, True
        return None, True
    best = None
    for st in part.compiled.stages:
        fs = set(st.formed_bend_uids)
        if set(formed) <= fs and not (fs & unformed):
            if best is None or st.stage_index < best.stage_index:
                best = st
    return (best.stage_model_id if best is not None else None), False
