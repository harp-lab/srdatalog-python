import sys,statistics,shutil
from driver import ROOT,command,load,save,gpu
from compare import compare
for ds in sys.argv[1:] or ('soot','jython','scala','kotlin'):
    records=[]
    for index in range(-1,3):
        for variant in (('baseline','field-flow') if index%2 else ('field-flow','baseline')):
            out=ROOT/'runs'/ds/f'{variant}-{index}.json';out.parent.mkdir(parents=True,exist_ok=True)
            execution=command([sys.executable,'-B',str(ROOT/'driver.py'),'run',ds,variant,str(out)],out.with_suffix('.log'))
            item=load(out);assert item['status']=='native_completed'
            records.append({'variant':variant,'index':index,'warmup':index<0,'fixedpoint_seconds':item['fixedpoint_seconds'],'process_seconds':execution['wall_seconds'],'report':str(out),'relation_counts':item['relation_counts']})
            save(ROOT/'results'/f'{ds}.json',{'status':'measuring','runs':records})
    first={v:next(r['relation_counts'] for r in records if r['variant']==v) for v in ('baseline','field-flow')}
    assert all(r['relation_counts']==first[r['variant']] for r in records)
    assert all(first['field-flow'][k]==n for k,n in first['baseline'].items()), 'original relation counts differ'
    medians={v:statistics.median(r['fixedpoint_seconds'] for r in records if r['variant']==v and not r['warmup']) for v in ('baseline','field-flow')}
    e2e={v:statistics.median(r['process_seconds'] for r in records if r['variant']==v and not r['warmup']) for v in medians}
    result={'status':'performance_complete','runs':records,'fixedpoint_medians_seconds':medians,'precompiled_process_wall_medians_seconds':e2e,'fixedpoint_reduction_percent':100*(1-medians['field-flow']/medians['baseline']),'process_reduction_percent':100*(1-e2e['field-flow']/e2e['baseline']),'timing_scope':'fresh Python driver process through normal exit; includes imports, binary hash verification, engine initialization, load, H2D, fixedpoint, cardinalities and teardown; excludes codegeneration, compilation and TSV exports','correctness':[]}
    save(ROOT/'results'/f'{ds}.json',result)
    for variant in ('field-flow',):
        folder=ROOT/'correctness'/ds/variant;folder.mkdir(parents=True,exist_ok=True);report=folder/'native.json';exports=folder/'relations'
        command([sys.executable,'-B',str(ROOT/'driver.py'),'run',ds,variant,str(report),str(exports)],folder/'native.log')
        native=load(report);assert native['status']=='native_completed'
        build=load(ROOT/'builds'/ds/variant/'build.json');build['split_applied']=True;build['string_array_type']=load(ROOT/'prepared'/ds/'meta.json')['java_lang_String_array']
        candidate={'status':'passed','dataset':ds,'variant':variant,'build':build,'expected_relations':[r['name'] for r in build['relations']],'input_relations':list(build['input_rows']),'outputs':native['outputs'],'relation_counts':native['relation_counts']}
        save(folder/'comparison-input.json',candidate)
        comparison=compare('/data/doop-corrected-xlarge/cpu/'+ds+'/result.json',folder/'comparison-input.json',ROOT/'prepared'/ds,folder/'comparison.json',ds,variant)
        assert comparison['passed']
        exports_info={k:{'path':v,'bytes':__import__('os').path.getsize(v),'sha256':gpu._sha256(__import__('pathlib').Path(v))} for k,v in native['outputs'].items()}
        save(folder/'export-identities.json',{'exports':exports_info,'retention':'removed after complete exact set checks to preserve 160GiB reserve; comparison sorted fingerprints retained'})
        shutil.rmtree(exports)
        result['correctness'].append({'variant':variant,'passed':True,'comparison':str(folder/'comparison.json')});save(ROOT/'results'/f'{ds}.json',result)
    result['status']='passed';save(ROOT/'results'/f'{ds}.json',result)
    print(ds,medians,result['fixedpoint_reduction_percent'],flush=True)
