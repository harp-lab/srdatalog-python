"""Exact original-IDB comparison with bounded scratch and factored HAS proof.

The 37 unchanged IDBs use complete external sorted sets, one relation at a time.
HAS is proved by exact nonstring equality plus String[] product cardinality,
uniqueness, and membership, without ever generating the Cartesian product.
SQL helper definitions are also checked in both EXCEPT directions. Requires
DuckDB for a split program; GNU sort/cmp for unchanged relations. Scratch is
released after each relation; CPU oracle and candidate exports are never deleted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from driver import gpu, load, ROOT, SOURCE


def string_array_type(facts):
    return load(Path(facts) / 'meta.json')['java_lang_String_array']

RESERVE = 160 * 1024 ** 3


def free_space(directory, extra=0):
    free = shutil.disk_usage(directory).free
    if free < RESERVE + extra:
        raise RuntimeError(f'Insufficient scratch space: {free} free; preserving {RESERVE} bytes')
    return free


def sorted_check(name, left, right, directory):
    paths = [Path(item['outputs'][name]).resolve(strict=True) for item in (left, right)]
    # Conservative room for both sorted copies and external-sort merge runs.
    free_space(directory, 3 * sum(path.stat().st_size for path in paths))
    with tempfile.TemporaryDirectory(prefix='.sets-', dir=directory) as temporary:
        scratch = Path(temporary)
        copies = [scratch / 'oracle', scratch / 'candidate']
        rows = []
        fingerprints = []
        for path, copy in zip(paths, copies):
            subprocess.run(['sort', '-u', '-S', '32G', '--parallel=8', '-T', str(scratch),
                            '-o', str(copy), str(path)],
                           env=dict(os.environ, LC_ALL='C'), check=True)
            count = 0
            with copy.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    count += chunk.count(b'\n')
            rows.append(count)
            fingerprints.append(gpu._sha256(copy))
        outcome = subprocess.run(['cmp', '-s', str(copies[0]), str(copies[1])])
        if outcome.returncode not in (0, 1):
            raise RuntimeError(f'cmp failed for {name}: {outcome.returncode}')
    expected = [item['relation_counts'][name] for item in (left, right)]
    return {'relation': name, 'method': 'Complete sorted unique integer TSV sets; no sampling',
            'oracle_rows': expected[0], 'candidate_rows': expected[1], 'unique_export_rows': rows,
            'sorted_unique_sha256': fingerprints,
            'equal_tuple_sets': outcome.returncode == 0,
            'passed': outcome.returncode == 0 and rows == expected and expected[0] == expected[1]}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def scan(path, arity):
    path = Path(path).resolve(strict=True)
    if path.stat().st_size == 0:
        return 'SELECT ' + ', '.join(f'CAST(NULL AS BIGINT) AS c{i}' for i in range(arity)) + ' WHERE FALSE'
    columns = ', '.join(f"'c{i}': 'BIGINT'" for i in range(arity))
    return (f'SELECT * FROM read_csv({literal(path)}, delim=\'\\t\', header=false, '
            f'auto_detect=false, columns={{{columns}}}, parallel=true)')


def helper_checks(left, right, facts, directory):
    import duckdb

    string_id = string_array_type(facts)
    if string_id is None or string_id != right['build']['string_array_type']:
        raise ValueError('Split helper metadata disagrees with String[] symbol')
    free = free_space(directory)
    with tempfile.TemporaryDirectory(prefix='.sql-', dir=directory) as temporary:
        connection = duckdb.connect()
        try:
            connection.execute("SET memory_limit='16GB'")
            connection.execute('SET threads=4')
            connection.execute('SET preserve_insertion_order=false')
            connection.execute(f'SET temp_directory={literal(temporary)}')
            spill = min(64 * 1024 ** 3, free - RESERVE)
            connection.execute(f"SET max_temp_directory_size='{spill}B'")

            def view(name, sql):
                connection.execute(f'CREATE TEMP VIEW {name} AS {sql}')

            def scalar(sql):
                return connection.execute(sql).fetchone()[0]

            def equality(name, actual, expected, arity, reported):
                columns = ', '.join(f'c{i}' for i in range(arity))
                rows = scalar(f'SELECT count(*) FROM ({actual})')
                unique = scalar(f'SELECT count(*) FROM (SELECT DISTINCT {columns} FROM ({actual}))')
                expected_rows = scalar(f'SELECT count(*) FROM (SELECT DISTINCT {columns} FROM ({expected}))')
                missing = scalar(f'SELECT count(*) FROM (({expected}) EXCEPT ({actual}))')
                extra = scalar(f'SELECT count(*) FROM (({actual}) EXCEPT ({expected}))')
                return {'relation': name, 'method': 'SQL EXCEPT in both directions; complete integer sets',
                        'export_rows': rows, 'unique_export_rows': unique, 'reported_rows': reported,
                        'expected_unique_rows': expected_rows, 'missing': missing, 'extra': extra,
                        'passed': missing == extra == 0 and rows == unique == reported == expected_rows}

            view('hat', scan(facts / 'HeapAllocation_Type.csv', 2))
            view('component', scan(facts / 'ComponentType.csv', 2))
            view('supertype', scan(left['outputs']['SupertypeOf'], 2))
            view('old_has', scan(left['outputs']['HeapAllocSuperType'], 2))
            # Needed for the disjoint decomposition, not assumed from Java types.
            multiple = scalar('SELECT count(*) FROM (SELECT c0 FROM hat GROUP BY c0 HAVING count(DISTINCT c1) > 1)')
            if multiple:
                raise ValueError('HAT is not functional: additive HAS decomposition is invalid; true union required')
            view('expected_arrays', f'SELECT DISTINCT c0 FROM hat WHERE c1 = {int(string_id)}')
            view('expected_eligible', f'''SELECT DISTINCT h.c0 FROM hat h
                JOIN supertype s ON s.c1 = h.c1
                JOIN component c ON c.c1 = s.c0 WHERE c.c0 = {int(string_id)}''')
            checks = []
            for name, definition in (('IsStringArrayHeap', 'SELECT * FROM expected_arrays'),
                                     ('EligibleStringArrayHeap', 'SELECT * FROM expected_eligible')):
                view(name, scan(right['outputs'][name], 1))
                checks.append(equality(name, f'SELECT * FROM {name}', definition, 1,
                                       right['relation_counts'][name]))
            view('nonstring', scan(right['outputs']['NonStringHeapAllocSuperType'], 2))
            expected_nonstring = '''SELECT o.* FROM old_has o WHERE NOT EXISTS
                (SELECT 1 FROM expected_arrays a WHERE a.c0 = o.c1)'''
            checks.append(equality('NonStringHeapAllocSuperType', 'SELECT * FROM nonstring',
                                   expected_nonstring, 2, right['relation_counts']['NonStringHeapAllocSuperType']))
            string_stats = connection.execute('''SELECT count(*), count(DISTINCT (o.c0, o.c1)),
                count(*) FILTER (WHERE NOT EXISTS
                    (SELECT 1 FROM expected_eligible e WHERE e.c0 = o.c0))
                FROM old_has o WHERE EXISTS
                    (SELECT 1 FROM expected_arrays a WHERE a.c0 = o.c1)''').fetchone()
            arrays = scalar('SELECT count(*) FROM expected_arrays')
            eligible = scalar('SELECT count(*) FROM expected_eligible')
            old_rows = scalar('SELECT count(*) FROM old_has')
            nonstring_rows = checks[-1]['expected_unique_rows']
            product = arrays * eligible
            proof = {
                'relation': 'HeapAllocSuperType',
                'method': 'Disjoint nonstring equality plus exact String[] subset membership, uniqueness and product cardinality; no Cartesian product materialized',
                'multiply_typed_heaps': multiple, 'oracle_export_rows': old_rows,
                'oracle_reported_rows': left['relation_counts']['HeapAllocSuperType'],
                'nonstring_rows': nonstring_rows, 'string_array_heaps': arrays,
                'eligible_heaps': eligible, 'logical_product_rows': product,
                'oracle_string_rows': string_stats[0], 'oracle_string_unique_rows': string_stats[1],
                'oracle_string_ineligible_pairs': string_stats[2],
                'logical_reconstructed_rows': nonstring_rows + product,
                'passed': (all(check['passed'] for check in checks)
                           and string_stats[0] == string_stats[1] == product and string_stats[2] == 0
                           and old_rows == left['relation_counts']['HeapAllocSuperType'] == nonstring_rows + product),
            }
            checks.append(proof)
            if right['variant'] == 'factor':
                view('store', scan(facts / 'StoreArrayIndex.csv', 3))
                view('source', scan(right['outputs']['StringArrayStoreSource'], 1))
                checks.append(equality('StringArrayStoreSource', 'SELECT * FROM source',
                                       'SELECT DISTINCT c0 FROM store', 1,
                                       right['relation_counts']['StringArrayStoreSource']))
                view('vpt', scan(right['outputs']['VarPointsTo'], 2))
                view('store_points', scan(right['outputs']['StringArrayStorePointsTo'], 2))
                definition = '''SELECT v.* FROM vpt v
                    WHERE EXISTS (SELECT 1 FROM store s WHERE s.c0 = v.c1)
                    AND EXISTS (SELECT 1 FROM expected_eligible e WHERE e.c0 = v.c0)'''
                checks.append(equality('StringArrayStorePointsTo', 'SELECT * FROM store_points',
                                       definition, 2, right['relation_counts']['StringArrayStorePointsTo']))
            return checks
        finally:
            connection.close()


def field_flow_checks(left, right, facts, directory):
    """Each field-flow helper equals its definition evaluated over the CPU oracle's VarPointsTo."""
    import duckdb
    free = free_space(directory)
    with tempfile.TemporaryDirectory(prefix='.sql-', dir=directory) as temporary:
        db = duckdb.connect()
        try:
            db.execute("SET memory_limit='48GB'")
            db.execute('SET threads=12')
            db.execute('SET preserve_insertion_order=false')
            db.execute(f'SET temp_directory={literal(temporary)}')
            db.execute(f"SET max_temp_directory_size='{min(64 * 1024 ** 3, free - RESERVE)}B'")
            for name, sql in (('vpt', scan(left['outputs']['VarPointsTo'], 2)),
                              ('reach', scan(left['outputs']['Reachable'], 1)),
                              ('rlif', scan(left['outputs']['ReachableLoadInstanceField'], 3)),
                              ('store', scan(facts / 'StoreInstanceField.csv', 4))):
                db.execute(f'CREATE TABLE {name} AS {sql}')
            sof = 'SELECT DISTINCT v.c0 AS c0, s.c2 AS c1, s.c0 AS c2 FROM store s JOIN reach r ON r.c0 = s.c3 JOIN vpt v ON v.c1 = s.c1'
            definitions = {
                'ReachableStoreInstanceField': (3, 'SELECT DISTINCT s.c1, s.c2, s.c0 FROM store s JOIN reach r ON r.c0 = s.c3'),
                'StoreObjFrom': (3, sof),
                'FieldFlow': (2, f'''SELECT DISTINCT o.c2, l.c2 FROM rlif l JOIN vpt v ON v.c1 = l.c0
                    JOIN ({sof}) o ON o.c0 = v.c0 AND o.c1 = l.c1'''),
            }
            checks = []
            for name, (arity, expected) in definitions.items():
                db.execute(f'CREATE TABLE expected AS {expected}')
                db.execute(f"CREATE TABLE actual AS {scan(right['outputs'][name], arity)}")

                def q(sql):
                    return db.execute(sql).fetchone()[0]
                columns = ', '.join(f'c{i}' for i in range(arity))
                rows = q('SELECT count(*) FROM actual')
                unique = q(f'SELECT count(*) FROM (SELECT DISTINCT {columns} FROM actual)')
                expected_rows = q('SELECT count(*) FROM expected')
                missing = q('SELECT count(*) FROM (SELECT * FROM expected EXCEPT SELECT * FROM actual)')
                extra = q('SELECT count(*) FROM (SELECT * FROM actual EXCEPT SELECT * FROM expected)')
                reported = right['relation_counts'][name]
                checks.append({'relation': name, 'method': 'Definition over CPU-oracle VarPointsTo; SQL EXCEPT both directions',
                               'export_rows': rows, 'unique_export_rows': unique, 'reported_rows': reported,
                               'expected_unique_rows': expected_rows, 'missing': missing, 'extra': extra,
                               'passed': missing == extra == 0 and rows == unique == reported == expected_rows})
                db.execute('DROP TABLE expected')
                db.execute('DROP TABLE actual')
            return checks
        finally:
            db.close()


def compare(oracle_path, candidate_path, facts, output, dataset, variant):
    left, right = load(oracle_path), load(candidate_path)
    if left.get('status') != 'passed' or right.get('status') != 'passed':
        raise ValueError('Only successful complete, process-gated runs may be compared')
    if left['dataset'] != dataset or right['dataset'] != dataset or right['variant'] != variant:
        raise ValueError('Dataset/variant mismatch')
    if left['source_sha256'] != '3b9c44856a0a469c48b832e03d8d8f78149a99a765cecb9c80d99cffaa162e94':
        raise ValueError('Oracle is not the corrected guarded query')
    if right['build']['source_sha256'] != gpu._sha256(SOURCE / 'examples/doop.py'):
        raise ValueError('Candidate is not frozen integrated07 query')
    old_facts = Path(left['facts'])
    old_manifest, new_manifest = load(old_facts / 'manifest.json'), load(facts / 'manifest.json')
    if left['input_manifest_sha256'] != gpu._sha256(old_facts / 'manifest.json'):
        raise ValueError('Oracle manifest changed')
    if old_manifest['source_sha256'] != new_manifest['source_sha256']:
        raise ValueError('Raw source facts differ')
    for name, entry in new_manifest['relations'].items():
        if entry['sha256'] != old_manifest['relations'][name]['sha256'] or entry['sha256'] != gpu._sha256(facts / entry['path']):
            raise ValueError('Prepared input differs: ' + name)
    old_meta, new_meta = load(old_facts / 'meta.json'), load(facts / 'meta.json')
    if any(new_meta.get(k) != v for k, v in old_meta.items()):
        raise ValueError('Shared metadata IDs differ')
    if right['build']['input_manifest_sha256'] != gpu._sha256(facts / 'manifest.json'):
        raise ValueError('Candidate fresh manifest changed')
    if right['build']['metadata_sha256'] != gpu._sha256(facts / 'meta.json'):
        raise ValueError('Candidate fresh metadata changed')
    inputs = set(left['input_relations'])
    original = set(left['expected_relations']) - inputs
    if set(left['outputs']) != original or set(left['relation_counts']) != set(left['expected_relations']):
        raise ValueError('CPU oracle does not cover every original relation')
    if inputs != set(right['input_relations']):
        raise ValueError('Input schemas differ')
    split = right['build'].get('split_applied', False)
    unchanged = original - {'HeapAllocSuperType'} if split else original
    helpers = {'NonStringHeapAllocSuperType', 'IsStringArrayHeap', 'EligibleStringArrayHeap'} if split else set()
    if split and variant == 'factor':
        helpers |= {'StringArrayStoreSource', 'StringArrayStorePointsTo'}
    field_flow = {'ReachableStoreInstanceField', 'StoreObjFrom', 'FieldFlow'}
    physical = unchanged | helpers | field_flow
    if set(right['outputs']) != physical or set(right['expected_relations']) != inputs | physical:
        raise ValueError('Candidate does not cover the complete expected physical query')
    if set(right['relation_counts']) != inputs | physical:
        raise ValueError('Candidate cardinality schema incomplete')
    if len(original) != 38 or (split and len(unchanged) != 37):
        raise ValueError('Unexpected canonical IDB schema; refusing a narrowed comparison')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    result = {
        'status': 'running', 'passed': False, 'dataset': dataset, 'variant': variant,
        'oracle_report': str(Path(oracle_path).resolve()),
        'candidate_report': str(Path(candidate_path).resolve()),
        'original_source_sha256': left['source_sha256'],
        'rewrite_sha256': right['build'].get('rewrite_sha256'),
        'program_identity_sha256': right['build'].get('program_identity_sha256'),
        'input_manifest_sha256': left['input_manifest_sha256'],
        'unchanged_original_idbs': sorted(unchanged), 'original_idb_count': len(original),
        'comparison_outside_fixedpoint_timing': True,
        'resources': {'sort_memory': '32G', 'sort_threads': 8, 'duckdb_memory': '16GB',
                      'duckdb_threads': 4, 'minimum_free_bytes': RESERVE,
                      'scratch_policy': 'One relation at a time; scratch deleted after comparison; exports retained'},
        'relations': [], 'helpers': [],
    }
    gpu._save(output, result)
    try:
        for name in sorted(inputs):
            a, b = left['relation_counts'][name], right['relation_counts'][name]
            result['relations'].append({'relation': name, 'method': 'Identical checksum-validated prepared inputs',
                                        'oracle_rows': a, 'candidate_rows': b, 'passed': a == b})
        for name in sorted(unchanged):
            result['relations'].append(sorted_check(name, left, right, output.parent))
            gpu._save(output, result)
        if split:
            result['helpers'] = helper_checks(left, right, facts, output.parent)
        result['helpers'] += field_flow_checks(left, right, facts, output.parent)
        result['passed'] = all(check['passed'] for check in result['relations'] + result['helpers'])
        result['status'] = 'passed' if result['passed'] else 'failed'
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        gpu._save(output, result)
    return result
