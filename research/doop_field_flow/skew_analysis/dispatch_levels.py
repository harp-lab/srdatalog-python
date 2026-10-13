import duckdb, json, sys, time
from pathlib import Path

# Throwaway: VirtualDispatch_MultiHead work units by binding level, var order
# [base, heap, invocation, simplename, descriptor, toMeth, thisP]. Final FULL x FULL.
WARPS = 1136 * 256 // 32
root = Path('/data/doop-corrected-xlarge')
for name in sys.argv[1:]:
  t0 = time.monotonic()
  folder = root / ('gpu-bitmap' if name == 'jython' else 'gpu-async') / name / 'relations'
  prepared = root / 'prepared' / name
  db = duckdb.connect()
  db.execute("SET threads=8; SET memory_limit='32GB'")
  def rel(path, cols):
    return ("read_csv('" + str(path) + "', delim='\\t', header=false, columns={"
            + ','.join("'" + c + "':'UINTEGER'" for c in cols) + "})")
  db.execute('CREATE TABLE ht AS SELECT * FROM ' + rel(prepared / 'HeapAllocation_Type.csv', ['heap', 'type']))
  db.execute('CREATE TABLE vmi AS SELECT v.* FROM ' + rel(folder / 'VirtualMethodInvocation.tsv', ['inv', 'base', 'sn', 'ds'])
             + ' v SEMI JOIN ' + rel(folder / 'ReachableInstruction.tsv', ['inv']) + ' r USING (inv)')
  db.execute('CREATE TABLE hh AS SELECT h.sn, h.ds, h.heap, h.thisp FROM '
             + rel(folder / 'HeapHelper.tsv', ['sn', 'ds', 'tometh', 'thisp', 'heap']) + ' h')
  # Resolution is a function of the heap's type: verify, then factor by type.
  db.execute('CREATE TABLE hht AS SELECT DISTINCT h.sn, h.ds, t.type, h.thisp FROM hh h JOIN ht t USING (heap)')
  per_type_unique = db.execute('SELECT count(*) = count(DISTINCT (sn, ds, type)) FROM hht').fetchone()[0]
  db.execute('CREATE TABLE pt AS SELECT v.var base, t.type, count(*)::BIGINT n FROM '
             + rel(folder / 'VarPointsTo.tsv', ['heap', 'var']) + ' v SEMI JOIN (SELECT DISTINCT base var FROM vmi) USING (var) '
             'JOIN ht t USING (heap) GROUP BY ALL')
  # Conditional-free unit: (base, inv, type) -> n heaps, each one output tuple.
  db.execute('CREATE TABLE u AS SELECT m.base, m.inv, p.type, h.thisp, p.n w FROM vmi m JOIN pt p USING (base) '
             'JOIN hht h ON h.sn = m.sn AND h.ds = m.ds AND h.type = p.type')
  def dist(q):
    n, tot, mx, med, p99 = db.execute('SELECT count(*), sum(w)::DOUBLE, max(w), median(w), quantile_cont(w, 0.99) FROM (' + q + ')').fetchone()
    return {'units': n, 'total': tot, 'max': mx, 'median': med, 'p99': p99, 'max_over_warp_fair_share': mx / (tot / WARPS)}
  res = {'dataset': name, 'resolution_unique_per_type': per_type_unique,
         'L0_base': dist('SELECT base, sum(w) w FROM u GROUP BY base'),
         'L_base_inv': dist('SELECT base, inv, sum(w) w FROM u GROUP BY ALL'),
         'L_base_inv_type': dist('SELECT w FROM u'),
         'by_thisP': dist('SELECT thisp, sum(w) w FROM u GROUP BY thisp'),
         'type_factored_rows': db.execute('SELECT count(*) FROM (SELECT DISTINCT base, inv, type FROM u)').fetchone()[0],
         'pt_base_type_rows': db.execute('SELECT count(*), sum(n) FROM pt').fetchone()}
  res['top_bases'] = [dict(zip(['base', 'w', 'invs', 'pts'], r)) for r in db.execute(
      'SELECT u.base, sum(w), count(DISTINCT inv), any_value(s.n) FROM u JOIN (SELECT base, sum(n) n FROM pt GROUP BY base) s USING (base) '
      'GROUP BY u.base ORDER BY 2 DESC LIMIT 8').fetchall()]
  res['seconds'] = time.monotonic() - t0
  print(json.dumps(res, indent=1, default=str), flush=True)
  db.close()
