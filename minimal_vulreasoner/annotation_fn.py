"""
PyReason annotation functions used by /reason.

Kept in its own module so the Docker warmup script (scripts/warmup_pyreason.py)
can import the exact same numba-compiled function the API invokes, without
dragging FastAPI / Pydantic / the rest of api.py through the warmup. Sharing
the function this way also keeps numba's first-class-function cache keys
identical between the warmup pass and the first real /reason call.
"""
import numba


# Per-grounding paired ann fn for analyst rules. Recovers the (Lcause, Leffect)
# pairing imposed by the connector clause (can_cause / contributes_to / derives /
# unsafe_variant_of / manifestation_of / implements). For each driver edge it
# looks up matching hasLabel(CB1, Lcause) and hasLabel(CB2, Leffect) bounds,
# takes their per-pair min (intersected with the connector's own bound), then
# returns the pair with the highest lower bound. The driver clause is detected
# dynamically as the edge clause whose two vars are both NOT head vars.
# analystAt(CB1) and stepFrom(CB1, CB2) are excluded from the per-pair min: they
# are gated by body thresholds upstream and equal [1,1] in the workflow.
@numba.njit
def paired_minimum_bounds_ann_fn(
    annotations, weights, qualified_nodes, qualified_edges, clause_labels, clause_variables
):
    head_var_x = "CB1"
    head_var_y = "CB2"

    driver_idx = -1
    has_label_cb1_idx = -1
    has_label_cb2_idx = -1

    for i in range(len(clause_labels)):
        name = clause_labels[i].value
        cv = clause_variables[i]
        if name == "hasLabel" and len(cv) == 2:
            if cv[0] == head_var_x:
                has_label_cb1_idx = i
            elif cv[0] == head_var_y:
                has_label_cb2_idx = i
        elif name != "analystAt" and name != "stepFrom":
            if (
                len(cv) == 2
                and cv[0] != head_var_x and cv[0] != head_var_y
                and cv[1] != head_var_x and cv[1] != head_var_y
            ):
                driver_idx = i

    if driver_idx < 0 or has_label_cb1_idx < 0 or has_label_cb2_idx < 0:
        return 0.0, 1.0

    best_lower = 0.0
    best_upper = 0.0
    found_any = False

    for ci in range(len(qualified_edges[driver_idx])):
        x_val = qualified_edges[driver_idx][ci][0]
        y_val = qualified_edges[driver_idx][ci][1]
        d_lower = annotations[driver_idx][ci].lower
        d_upper = annotations[driver_idx][ci].upper

        x_lower = -1.0
        x_upper = -1.0
        for k in range(len(qualified_edges[has_label_cb1_idx])):
            if qualified_edges[has_label_cb1_idx][k][1] == x_val:
                x_lower = annotations[has_label_cb1_idx][k].lower
                x_upper = annotations[has_label_cb1_idx][k].upper
                break
        if x_lower < 0.0:
            continue

        y_lower = -1.0
        y_upper = -1.0
        for k in range(len(qualified_edges[has_label_cb2_idx])):
            if qualified_edges[has_label_cb2_idx][k][1] == y_val:
                y_lower = annotations[has_label_cb2_idx][k].lower
                y_upper = annotations[has_label_cb2_idx][k].upper
                break
        if y_lower < 0.0:
            continue

        pair_lower = min(min(x_lower, y_lower), d_lower)
        pair_upper = min(min(x_upper, y_upper), d_upper)

        if not found_any or pair_lower > best_lower:
            best_lower = pair_lower
            best_upper = pair_upper
            found_any = True

    if not found_any:
        return 0.0, 1.0
    lower = min(best_lower, 1.0)
    upper = min(best_upper, 1.0)
    if lower > upper:
        return 0.0, 1.0
    return lower, upper
