import sqlite3,json,re,collections,sys
from pathlib import Path

def union(intervals):
    total=0;left=right=None
    for a,b in sorted(intervals):
        if left is None:left,right=a,b
        elif a>right:total+=right-left;left,right=a,b
        else:right=max(right,b)
    return total+(right-left if left is not None else 0)

def analyze(path):
    db=sqlite3.connect(path);db.row_factory=sqlite3.Row
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    kernels=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
    runtime=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start')]
    ranges=[dict(r) for r in db.execute('SELECT * FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start')]
    bythread=collections.defaultdict(list)
    for r in ranges:bythread[r['globalTid']].append(r)
    positions=collections.defaultdict(int);active=collections.defaultdict(list);labels={};api=collections.defaultdict(list)
    for r in runtime:
        tid=r['globalTid'];seq=bythread[tid];p=positions[tid];a=r['start'];b=r['end']
        live=[v for v in active[tid] if v['end']>=a]
        while p<len(seq) and seq[p]['start']<=a:
            if seq[p]['end']>=a:live.append(seq[p])
            p+=1
        positions[tid]=p;active[tid]=live
        stack=[v.get('text') or strings.get(v.get('textId'),'') for v in live if v['end']>=b]
        labels[r['correlationId']]=stack
        name=strings[r['nameId']];api[name].append((a,b))
    rule=collections.defaultdict(list);families=collections.defaultdict(list);names=collections.defaultdict(list);phases=collections.defaultdict(list)
    for k in kernels:
        name=strings[k['demangledName']];interval=(k['start'],k['end']);names[name].append(k)
        m=re.search(r'JitRunner_(\w+)::kernel_(\w+)',name)
        stack=labels.get(k['correlationId'],[])
        if m:
            label=m.group(1);phase=m.group(2);rule[label].append(k);families[re.sub(r'_D\d+$','',label)].append(k);category='join_'+phase
        elif 'bitmap' in name.lower():category='bitmap';rule['VPT_Assign_bitmap'].append(k);families['VPT_Assign'].append(k)
        elif any('merge' in s.lower() for s in stack):category='merge'
        elif any('sort' in s.lower() for s in stack) or any(s in name for s in ('Sort','sort','Radix')):category='sort'
        elif any('Index' in s or 'Dedup' in s for s in stack):category='index_dedup'
        elif 'scan' in name.lower():category='scan'
        else:category='other'
        phases[category].append(interval)
    intervals=[(k['start'],k['end']) for k in kernels];busy=union(intervals);span=max(b for a,b in intervals)-min(a for a,b in intervals)
    def ranked(groups):
        result=[]
        for name,items in groups.items():
            durations=[k['end']-k['start'] for k in items];idx=max(range(len(items)),key=durations.__getitem__);top=items[idx]
            result.append({'name':name,'launches':len(items),'kernel_sum_seconds':sum(durations)/1e9,'kernel_union_seconds':union([(k['start'],k['end']) for k in items])/1e9,'max_launch_seconds':durations[idx]/1e9,'longest_launch_matching_skip':idx,'longest_kernel':strings[top['demangledName']],'longest_registers_per_thread':top['registersPerThread'],'longest_grid_x':top['gridX'],'longest_block_x':top['blockX'],'phase_sums':dict(collections.Counter({p:sum(k['end']-k['start'] for k in items if p in strings[k['demangledName']])/1e9 for p in ('kernel_count','kernel_materialize','kernel_bg_count','kernel_bg_materialize','kernel_bg_histogram')}))})
        return sorted(result,key=lambda r:-r['kernel_sum_seconds'])
    result={'database':str(path),'scope':'cudaProfilerStart/Stop around unlimited fixedpoint + synchronization; profiling is not timing evidence','gpu_kernel_span_seconds':span/1e9,'gpu_kernel_union_seconds':busy/1e9,'kernel_sum_seconds':sum(b-a for a,b in intervals)/1e9,'overlap_excess_seconds':(sum(b-a for a,b in intervals)-busy)/1e9,'gap_within_kernel_span_seconds':(span-busy)/1e9,'nonadditivity':'Kernel duration sums overlap across streams, phases/rule unions overlap; do not sum as elapsed contributions. Runtime API time may overlap kernels.','rule_variants':ranked(rule),'rule_families':ranked(families),'kernels':ranked(names),'phases':{k:{'sum_seconds':sum(b-a for a,b in v)/1e9,'union_seconds':union(v)/1e9,'launches':len(v)} for k,v in phases.items()},'runtime_apis':sorted([{'name':k,'calls':len(v),'sum_seconds':sum(b-a for a,b in v)/1e9,'union_seconds':union(v)/1e9} for k,v in api.items()],key=lambda x:-x['sum_seconds'])}
    db.close();return result
if __name__=='__main__':
    path=Path(sys.argv[1]);out=path.with_suffix('.analysis.json');out.write_text(json.dumps(analyze(path),indent=2)+'\n');print(out)
