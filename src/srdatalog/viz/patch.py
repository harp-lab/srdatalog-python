'''Patch a rule's `.with_plan(...)` kwargs in a source file.

Strategy: use `find_rule_locations` to get the byte-offset span of each
`with_plan(var_order=VALUE, clause_order=VALUE)` kwarg value, then slice
+ reassemble the source. This preserves surrounding formatting — the
user's quote style, line breaks, comments outside the kwarg VALUE all
stay intact.

For kwargs we want to introduce that don't exist yet (e.g. the rule
has `.with_plan(var_order=[...])` and we want to add `clause_order=[...]`
too), we append them inside the existing `.with_plan(...)` call's
paren range. For rules that have no `.with_plan(...)` at all, we
append one after the `.named(...)` call.

Not a general refactoring tool — intended specifically for the viz
extension's "drag reorder → write back" loop.
'''

from __future__ import annotations

from dataclasses import dataclass

from srdatalog.viz.source import (
  PlanCallSpan,
  RuleLocation,
  find_rule_locations,
)


class PlanPatchError(ValueError):
  '''Raised when the source doesn't contain the target rule, or the
  rule has no `.named(...)` call to patch against.'''


@dataclass
class _Edit:
  start: int
  end: int
  new: str


def patch_rule_plan(
  source: str,
  rule_name: str,
  *,
  var_order: list[str] | None = None,
  clause_order: list[int] | None = None,
  delta: int = -1,
) -> str:
  '''Return `source` with rule `rule_name`'s plan updated.

  Args:
    source: full source text.
    rule_name: the string inside `.named("...")`.
    var_order: new variable order. None = leave unchanged.
    clause_order: new clause index order. None = leave unchanged.
    delta: which PlanEntry to target (when multiple .with_plan chained).
      -1 = the first one / append a new one. Non-negative values match
      an existing entry with `delta=N`, if present.

  At least one of `var_order` / `clause_order` must be non-None.
  '''
  if var_order is None and clause_order is None:
    raise PlanPatchError("patch_rule_plan: pass var_order and/or clause_order")

  locs = find_rule_locations(source)
  target = next((l for l in locs if l.name == rule_name), None)
  if target is None:
    names = ", ".join(l.name for l in locs) or "<none>"
    raise PlanPatchError(f"rule {rule_name!r} not found in source. Rules seen: {names}")

  if target.plan_calls:
    # Pick the with_plan call whose `delta` kwarg matches (or the first
    # one if delta=-1 and no explicit match).
    call = _select_plan_call(target.plan_calls, source, delta)
    if call is not None:
      return _edit_plan_call(source, call, var_order, clause_order, delta)

  # No plan calls — append a new `.with_plan(...)` after `.named("X")`.
  return _append_plan_call(source, target, var_order, clause_order, delta)


# ---------------------------------------------------------------------------
# Existing-call edit
# ---------------------------------------------------------------------------


def _select_plan_call(calls: list[PlanCallSpan], source: str, delta: int) -> PlanCallSpan | None:
  '''Find the PlanCall whose `delta` kwarg equals `delta`. If delta=-1
  and no call has explicit delta, fall back to the first call.'''
  for c in calls:
    call_delta = _read_kwarg_int(c, source, "delta", default=-1)
    if call_delta == delta:
      return c
  return calls[0] if delta == -1 else None


def _edit_plan_call(
  source: str,
  call: PlanCallSpan,
  var_order: list[str] | None,
  clause_order: list[int] | None,
  delta: int,
) -> str:
  '''Apply kwarg edits to an existing with_plan call.'''
  edits: list[_Edit] = []
  existing_kwargs = {kw.kwarg: kw for kw in call.kwargs}

  if var_order is not None:
    edits.append(
      _kwarg_edit(call, existing_kwargs, source, "var_order", _format_str_list(var_order))
    )
  if clause_order is not None:
    edits.append(
      _kwarg_edit(call, existing_kwargs, source, "clause_order", _format_int_list(clause_order))
    )

  # delta propagation: if user asked for a specific delta (non-default)
  # and the call doesn't already carry it, add it too. Silently skip
  # when delta is the default (-1).
  if delta != -1 and "delta" not in existing_kwargs:
    edits.append(_kwarg_edit(call, existing_kwargs, source, "delta", str(delta)))

  return _apply_edits(source, edits)


def _kwarg_edit(
  call: PlanCallSpan,
  existing: dict,
  source: str,
  name: str,
  new_value_text: str,
) -> _Edit:
  if name in existing:
    span = existing[name]
    return _Edit(start=span.start, end=span.end, new=new_value_text)
  # Not present — insert at end of call's arg list. Find the `)`.
  return _insert_new_kwarg_edit(source, call, name, new_value_text)


def _insert_new_kwarg_edit(
  source: str, call: PlanCallSpan, name: str, new_value_text: str
) -> _Edit:
  '''Append `, name=VALUE` just before the closing `)` of the call.'''
  # Walk back from call.end - 1 (which is the `)`, since end is
  # exclusive) to find the last non-whitespace character.
  close_paren = call.end - 1
  while close_paren > call.start and source[close_paren] != ")":
    close_paren -= 1
  if source[close_paren] != ")":
    raise PlanPatchError(f"malformed with_plan(...) call at offset {call.start}")

  # Insert `, name=VALUE` right before `)`. Check whether there's
  # already a trailing comma; if yes, omit our leading comma.
  # Simple heuristic: scan back past whitespace/newlines.
  scan = close_paren - 1
  while scan > call.start and source[scan] in " \t\n":
    scan -= 1
  needs_leading_comma = scan > call.start and source[scan] != ","
  # Check if the call has any args at all — scan for an opening paren
  # then check if non-whitespace follows before the close.
  has_args = _call_has_args(source, call)
  sep = ", " if (has_args and needs_leading_comma) else ""
  insertion = f"{sep}{name}={new_value_text}"
  return _Edit(start=close_paren, end=close_paren, new=insertion)


def _call_has_args(source: str, call: PlanCallSpan) -> bool:
  '''True if `.with_plan(...)` has at least one arg/kwarg.'''
  open_paren = source.find("(", call.start)
  if open_paren == -1:
    return False
  close = call.end - 1
  body = source[open_paren + 1 : close].strip()
  return bool(body)


# ---------------------------------------------------------------------------
# Append a new with_plan() when none exists
# ---------------------------------------------------------------------------


def _append_plan_call(
  source: str,
  target: RuleLocation,
  var_order: list[str] | None,
  clause_order: list[int] | None,
  delta: int,
) -> str:
  '''Insert `.with_plan(...)` right after the rule's `.named("X")` call.

  We walk forward from target.start looking for `.named("name")` and
  splice a new call in after its `)`.
  '''
  marker = f'.named("{target.name}")'
  idx = source.find(marker, target.start)
  if idx == -1:
    # Try single quotes
    marker = f".named('{target.name}')"
    idx = source.find(marker, target.start)
  if idx == -1:
    raise PlanPatchError(f"could not locate .named({target.name!r}) call")
  insert_at = idx + len(marker)
  parts: list[str] = []
  if delta != -1:
    parts.append(f"delta={delta}")
  if var_order is not None:
    parts.append(f"var_order={_format_str_list(var_order)}")
  if clause_order is not None:
    parts.append(f"clause_order={_format_int_list(clause_order)}")
  call_text = f".with_plan({', '.join(parts)})"
  return source[:insert_at] + call_text + source[insert_at:]


# ---------------------------------------------------------------------------
# Edit application + helpers
# ---------------------------------------------------------------------------


def _apply_edits(source: str, edits: list[_Edit]) -> str:
  '''Apply non-overlapping edits in descending start order.'''
  for e in sorted(edits, key=lambda e: e.start, reverse=True):
    source = source[: e.start] + e.new + source[e.end :]
  return source


def _format_str_list(xs: list[str]) -> str:
  return "[" + ", ".join(f'"{x}"' for x in xs) + "]"


def _format_int_list(xs: list[int]) -> str:
  return "[" + ", ".join(str(x) for x in xs) + "]"


def _read_kwarg_int(call: PlanCallSpan, source: str, name: str, default: int) -> int:
  for kw in call.kwargs:
    if kw.kwarg == name:
      try:
        return int(source[kw.start : kw.end].strip())
      except ValueError:
        return default
  return default
