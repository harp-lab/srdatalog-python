import duckdb, json, sys, time
from pathlib import Path

# Throwaway: VPT_LoadField work units at successive binding levels of the
# generated var order [baseheap, sig, base, heap, to], plus exact set sharing.
# Final FULL x FULL structure, not per-iteration DELTA work.
WARPS = 1136 * 256 // 32  # grid used by the profiled launches
root = Path('/data/doop-corrected-xlarge')
for name in sys.argv[1:]:
  t0 = time.monotonic()
  folder = root / ('gpu-bitmap' if name == 'jython' else 'gpu-async') / name / 'relations'
  db = duckdb.connect()
  db.execute("SET threads=12; SET memory_limit='48GB'")
  rel = lambda f, cols: ("read_csv('" + str(folder / f) + "', delim='\\t', header=false, columns={"
                         + ','.join("'" + c + "':'UINTEGER'" for c in cols) + "})")
  db.execute('CREATE TABLE rlif AS SELECT * FROM ' + rel('ReachableLoadInstanceField.tsv', ['base', 'sig', 'dest']))
  db.execute('CREATE TABLE loads AS SELECT base, sig, count(*)::BIGINT d FROM rlif GROUP BY ALL')
  db.execute('CREATE TABLE ifpt AS SELECT * FROM ' + rel('InstanceFieldPointsTo.tsv', ['heap', 'sig', 'bh']))
  db.execute('CREATE TABLE cells AS SELECT bh, sig, count(*)::BIGINT h, hash(list(heap ORDER BY heap)) hs FROM ifpt GROUP BY ALL')
  db.execute('CREATE TABLE alias AS SELECT v.bh, v.base FROM ' + rel('VarPointsTo.tsv', ['bh', 'base'])
             + ' v SEMI JOIN (SELECT DISTINCT base FROM loads) USING (base)')
  # L2 units: (bh, sig, base); residual = heaps(bh,sig) x dests(base,sig): conditional-free Cartesian.
  db.execute('CREATE TABLE l2 AS SELECT a.bh, l.sig, a.base, l.d, c.h, c.hs, (l.d * c.h)::BIGINT w '
             'FROM alias a JOIN loads l USING (base) JOIN cells c ON c.bh = a.bh AND c.sig = l.sig')

  def dist(q):
    r = db.execute('SELECT count(*), sum(w)::DOUBLE, max(w), median(w), quantile_cont(w, 0.99) FROM (' + q + ')').fetchone()
    n, tot, mx, med, p99 = r
    return {'units': n, 'total': tot, 'max': mx, 'median': med, 'p99': p99,
            'max_over_warp_fair_share': mx / (tot / WARPS)}

  res = {'dataset': name, 'warps': WARPS,
         'L0_baseheap': dist('SELECT bh, sum(w) w FROM l2 GROUP BY bh'),
         'L1_baseheap_sig': dist('SELECT bh, sig, sum(w) w FROM l2 GROUP BY bh, sig'),
         'L2_baseheap_sig_base': dist('SELECT w FROM l2')}
  # Exact sharing: cells with identical field content set; and identical (content, loader-destination) pairs,
  # whose derivations are byte-identical duplicates across baseheaps.
  res['cells'] = dict(zip(['cells', 'distinct_content_sets', 'ifpt_rows', 'rows_if_sets_shared'],
                          db.execute('SELECT count(*), count(DISTINCT (sig, hs)), sum(h), '
                                     '(SELECT sum(h) FROM (SELECT DISTINCT sig, hs, h FROM cells)) FROM cells').fetchone()))
  db.execute('CREATE TABLE dests AS SELECT a.bh, r.sig, hash(list(DISTINCT r.dest ORDER BY r.dest)) ds '
             'FROM alias a JOIN rlif r USING (base) GROUP BY ALL')
  db.execute('CREATE TABLE l1 AS SELECT bh, sig, any_value(hs) hs, sum(w) w FROM l2 GROUP BY bh, sig')
  tot, uniq = db.execute('SELECT sum(w)::DOUBLE, (SELECT sum(w)::DOUBLE FROM (SELECT DISTINCT l1.sig, hs, ds, w '
                         'FROM l1 JOIN dests USING (bh, sig))) FROM l1').fetchone()
  res['duplicate_derivations'] = {'total': tot, 'after_memoizing_identical_content_and_dest_sets': uniq,
                                  'redundant_fraction': 1 - uniq / tot}
  res['seconds'] = time.monotonic() - t0
  print(json.dumps(res, indent=1), flush=True)
  db.close()
