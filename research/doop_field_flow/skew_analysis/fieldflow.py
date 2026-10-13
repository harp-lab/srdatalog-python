import duckdb, json, sys, time
from pathlib import Path

# Throwaway: predict the upstream-Doop field formulation on final FULL relations.
#   SOF(o,f,from)   <- Reachable(m), StoreInstanceField(from,base,f,m), VPT(o,base)
#   FF(to,from)     <- RLIF(base,f,to), VPT(o,base), SOF(o,f,from)
#   VPT(h,to)       <- FF(to,from), VPT(h,from)        (root key `from`: work = outdeg x |pts|)
WARPS = 1136 * 256 // 32
root = Path('/data/doop-corrected-xlarge')
for name in sys.argv[1:]:
  t0 = time.monotonic()
  folder = root / ('gpu-bitmap' if name == 'jython' else 'gpu-async') / name / 'relations'
  prepared = root / 'prepared' / name
  db = duckdb.connect()
  db.execute("SET threads=12; SET memory_limit='48GB'")
  def rel(path, cols):
    return ("read_csv('" + str(path) + "', delim='\\t', header=false, columns={"
            + ','.join("'" + c + "':'UINTEGER'" for c in cols) + "})")
  db.execute('CREATE TABLE rlif AS SELECT * FROM ' + rel(folder / 'ReachableLoadInstanceField.tsv', ['base', 'sig', 'dest']))
  db.execute('CREATE TABLE rsif AS SELECT DISTINCT s.frm, s.base, s.sig FROM '
             + rel(prepared / 'StoreInstanceField.csv', ['frm', 'base', 'sig', 'm']) + ' s SEMI JOIN '
             + rel(folder / 'Reachable.tsv', ['m']) + ' r USING (m)')
  db.execute('CREATE TABLE vpt AS SELECT * FROM ' + rel(folder / 'VarPointsTo.tsv', ['h', 'v']))
  db.execute('CREATE TABLE sizes AS SELECT v, count(*)::BIGINT n FROM vpt GROUP BY v')
  db.execute('CREATE TABLE sof AS SELECT DISTINCT p.h o, s.sig, s.frm FROM rsif s JOIN vpt p ON p.v = s.base')
  # FF derivation witnesses grouped by (o,f): loads aliasing o.f x stores into o.f.
  db.execute('CREATE TABLE ld AS SELECT p.h o, l.sig, count(*)::BIGINT nl FROM rlif l JOIN vpt p ON p.v = l.base GROUP BY ALL')
  db.execute('CREATE TABLE st AS SELECT o, sig, count(*)::BIGINT ns FROM sof GROUP BY ALL')
  ffw = db.execute('SELECT count(*), sum(nl * ns)::DOUBLE, max(nl * ns) FROM ld JOIN st USING (o, sig)').fetchone()
  db.execute('CREATE TABLE ff AS SELECT DISTINCT l.dest "to", s.frm "from" FROM rlif l JOIN vpt p ON p.v = l.base '
             'JOIN sof s ON s.o = p.h AND s.sig = l.sig')
  def dist(q):
    n, tot, mx, med, p99 = db.execute('SELECT count(*), sum(w)::DOUBLE, max(w), median(w), quantile_cont(w, 0.99) FROM (' + q + ')').fetchone()
    return {'units': n, 'total': tot, 'max': mx, 'median': med, 'p99': p99, 'max_over_warp_fair_share': mx / (tot / WARPS)}
  res = {'dataset': name,
         'sof_rows': db.execute('SELECT count(*) FROM sof').fetchone()[0],
         'ff_derivation': {'o_f_units': ffw[0], 'witnesses': ffw[1], 'max_unit': ffw[2]},
         'ff_edges': db.execute('SELECT count(*) FROM ff').fetchone()[0],
         'propagation_by_from': dist('SELECT f."from", (count(*) * any_value(z.n))::BIGINT w FROM ff f JOIN sizes z ON z.v = f."from" GROUP BY f."from"'),
         'propagation_by_edge': dist('SELECT z.n w FROM ff f JOIN sizes z ON z.v = f."from"'),
         'ff_derivation_by_o_f': dist('SELECT nl * ns w FROM ld JOIN st USING (o, sig)')}
  res['seconds'] = time.monotonic() - t0
  print(json.dumps(res, indent=1), flush=True)
  db.close()
