'''galen.nim -- 8 rules incl 3-way joins with plan pragmas'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import PlanEntry, Program, Relation, Var


def build_galen() -> Program:
  X, Y, Z, R, W, U, E, O, Q_V = (
    Var("x"),
    Var("y"),
    Var("z"),
    Var("r"),
    Var("w"),
    Var("u"),
    Var("e"),
    Var("o"),
    Var("q"),
  )
  p_in = Relation("PInput", 2)
  q_in = Relation("QInput", 3)
  r_in = Relation("RInput", 3)
  c_in = Relation("CInput", 3)
  u_in = Relation("UInput", 3)
  s_in = Relation("SInput", 2)
  outp = Relation("OutP", 2)
  outq = Relation("OutQ", 3)
  return Program(
    relations=[p_in, q_in, r_in, c_in, u_in, s_in, outp, outq],
    rules=[
      (outp(X, Z) <= p_in(X, Z)).named("LoadP"),
      (outq(X, R, Z) <= q_in(X, R, Z)).named("LoadQ"),
      (outp(X, Z) <= outp(X, Y) & outp(Y, Z)).named("TC"),
      (outq(X, R, Z) <= outp(X, Y) & outq(Y, R, Z)).named("PropQ"),
      (outp(X, Z) <= outp(Y, W) & u_in(W, R, Z) & outq(X, R, Y))
      .named("Join3a")
      .with_plans(
        [
          PlanEntry(delta=2, var_order=("y", "w", "r", "x", "z")),
        ]
      ),
      (outp(X, Z) <= c_in(Y, W, Z) & outp(X, W) & outp(X, Y)).named("Join3b"),
      (outq(X, Q_V, Z) <= outq(X, R, Z) & s_in(R, Q_V)).named("PropQS"),
      (outq(X, E, O) <= outq(X, Y, Z) & r_in(Y, U, E) & outq(Z, U, O))
      .named("Join3c")
      .with_plans(
        [
          PlanEntry(delta=0, var_order=("z", "y", "u", "x", "e", "o")),
        ]
      ),
    ],
  )


def test_galen_hir():
  diff_hir(build_galen(), "galen")


def test_galen_mir():
  diff_mir(build_galen(), "galen")


if __name__ == "__main__":
  test_galen_hir()
  test_galen_mir()
  print("galen: OK")
