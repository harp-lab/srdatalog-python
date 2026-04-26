'''AST-based rule location extraction.

Walks a Python source file looking for the pattern:

    (HEAD_ATOM <= BODY).named("RuleName")[.with_plan(kwargs...)]*

and produces a mapping from rule name → source location, plus, for
each `.with_plan(...)` call, the exact byte-offset range of each
keyword argument's value. The extension uses this to:

  1. Jump to a rule from the visualization sidebar
  2. Replace a specific `var_order=[...]` / `clause_order=[...]` value
     without reformatting the rest of the file

We key rules by the string literal passed to `.named(...)` because
that's the only name the HIR / JIT stages know them by. Anonymous
rules (no `.named` call) are skipped — the extension has no handle
for them anyway.
'''

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanKwargSpan:
  '''Byte-offset range of one `with_plan(kwarg=VALUE)` — VALUE only.'''

  kwarg: str  # e.g. "var_order", "clause_order"
  start: int  # inclusive, 0-based byte offset into source
  end: int  # exclusive


@dataclass(frozen=True)
class PlanCallSpan:
  '''Byte-offset range of one `.with_plan(...)` call on a rule.'''

  start: int
  end: int
  kwargs: list[PlanKwargSpan] = field(default_factory=list)


@dataclass(frozen=True)
class RuleLocation:
  '''Where a named rule lives in the source + what plans it has.'''

  name: str
  start_line: int  # 1-based
  end_line: int  # 1-based, inclusive
  start: int  # byte offset of the outermost expression
  end: int
  plan_calls: list[PlanCallSpan] = field(default_factory=list)


def find_rule_locations(source: str) -> list[RuleLocation]:
  '''Walk the AST, return one RuleLocation per `.named("X")`-suffixed rule.

  The pattern we recognize:
      ( HEAD <= BODY ).named("NAME").with_plan(kw=VAL, ...) ...

  We accept any number (including zero) of trailing `.with_plan(...)`
  calls after `.named(...)`. Rules without `.named` are skipped.
  '''
  tree = ast.parse(source)
  lines_offsets = _line_offsets(source)
  locs: list[RuleLocation] = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    name = _named_call_name(node)
    if name is None:
      continue
    # Peel back to the outermost expression — keep extending while the
    # parent is `.with_plan(...)` so we capture all trailing calls.
    outer = node
    plan_calls: list[PlanCallSpan] = []
    # The AST builds chains bottom-up: outermost call is what .walk
    # hits LAST for a given chain. Instead of peeling, we detect all
    # `.with_plan` Call nodes whose .func is an Attribute rooted at
    # this `.named(...)` — safer to run a targeted traversal from the
    # chain's outermost ancestor. Simpler: a second pass below.
    locs.append(
      RuleLocation(
        name=name,
        start_line=outer.lineno,
        end_line=outer.end_lineno or outer.lineno,
        start=_offset(lines_offsets, outer.lineno, outer.col_offset),
        end=_offset(lines_offsets, outer.end_lineno or outer.lineno, outer.end_col_offset or 0),
        plan_calls=plan_calls,
      )
    )

  # Second pass: for each rule, find any `.with_plan(...)` calls whose
  # chain descends through its `.named(...)` call.
  locs_by_offset = {(l.start_line, l.name): l for l in locs}
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    rule_name = _with_plan_chain_owner(node)
    if rule_name is None:
      continue
    key = (_chain_named_line(node), rule_name)
    target = locs_by_offset.get(key)
    if target is None:
      continue
    span = _plan_call_span(node, source, lines_offsets)
    target.plan_calls.append(span)
    # Extend the rule's overall range to cover the trailing call.
    new_end_line = node.end_lineno or target.end_line
    new_end = _offset(lines_offsets, new_end_line, node.end_col_offset or 0)
    if new_end > target.end:
      # RuleLocation is frozen, so replace in-place via dataclasses trick.
      object.__setattr__(target, "end_line", new_end_line)
      object.__setattr__(target, "end", new_end)

  # Sort by line for deterministic output + stable editor jumps.
  locs.sort(key=lambda l: (l.start_line, l.name))
  return locs


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _named_call_name(call: ast.Call) -> str | None:
  '''If `call` is `X.named("string")`, return the string. Else None.'''
  func = call.func
  if not isinstance(func, ast.Attribute):
    return None
  if func.attr != "named":
    return None
  if len(call.args) != 1:
    return None
  arg = call.args[0]
  if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
    return arg.value
  return None


def _with_plan_chain_owner(call: ast.Call) -> str | None:
  '''If `call` is `X.with_plan(...)` with a `.named("name")` upstream in
  the chain, return the name. Else None.'''
  func = call.func
  if not isinstance(func, ast.Attribute) or func.attr != "with_plan":
    return None
  cur: ast.AST | None = func.value
  while isinstance(cur, (ast.Call, ast.Attribute)):
    if isinstance(cur, ast.Call):
      name = _named_call_name(cur)
      if name is not None:
        return name
      cur = cur.func
    else:
      cur = cur.value
  return None


def _chain_named_line(call: ast.Call) -> int:
  '''The line of the `.named(...)` call in this with_plan chain.'''
  cur: ast.AST | None = call.func
  while isinstance(cur, (ast.Call, ast.Attribute)):
    if isinstance(cur, ast.Call) and _named_call_name(cur) is not None:
      return cur.lineno
    cur = cur.func if isinstance(cur, ast.Call) else cur.value
  return call.lineno


def _plan_call_span(call: ast.Call, source: str, line_offsets: list[int]) -> PlanCallSpan:
  start = _offset(line_offsets, call.lineno, call.col_offset)
  end = _offset(line_offsets, call.end_lineno or call.lineno, call.end_col_offset or 0)
  kwargs: list[PlanKwargSpan] = []
  for kw in call.keywords:
    if kw.arg is None:
      continue
    v = kw.value
    ks = _offset(line_offsets, v.lineno, v.col_offset)
    ke = _offset(line_offsets, v.end_lineno or v.lineno, v.end_col_offset or 0)
    kwargs.append(PlanKwargSpan(kwarg=kw.arg, start=ks, end=ke))
  return PlanCallSpan(start=start, end=end, kwargs=kwargs)


def _line_offsets(source: str) -> list[int]:
  '''1-based line → starting byte offset (0-based). Index 0 unused.'''
  offsets = [0, 0]  # offsets[1] = 0 (first line starts at 0)
  off = 0
  for line in source.splitlines(keepends=True):
    off += len(line)
    offsets.append(off)
  return offsets


def _offset(line_offsets: list[int], line: int, col: int) -> int:
  return line_offsets[line] + col
