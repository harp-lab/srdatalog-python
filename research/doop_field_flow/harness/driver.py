import ctypes, json, os, sys, time, hashlib, subprocess, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
sys.path[:0]=[str(SOURCE/'src'),str(SOURCE/'examples')]
os.environ['SRDATALOG_RMM_RESOURCE']='cuda_async'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from doop_suite import gpu

def load(p): return json.loads(Path(p).read_text())
def save(p,v): Path(p).parent.mkdir(parents=True,exist_ok=True); gpu._save(Path(p),v)
def guard():
    assert shutil.disk_usage(ROOT).free > 160*1024**3
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if apps: raise RuntimeError('Foreign GPU process: '+apps)
def command(args,log):
    guard(); log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
    record={'argv':list(map(str,args)),'cwd':str(ROOT),'start':time.time(),'free_bytes':shutil.disk_usage(ROOT).free}
    with log.open('x') as f: p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    record.update(returncode=p.returncode,wall_seconds=time.time()-record['start'])
    with (ROOT/'commands.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    if p.returncode: raise RuntimeError(str(log))
    return record

def profile(ds,variant,out):
    build=load(ROOT/'builds'/ds/variant/'build.json'); facts=ROOT/'prepared'/ds
    lib=ctypes.CDLL(build['library'],mode=ctypes.RTLD_GLOBAL)
    cuda=ctypes.CDLL('/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib/libcudart.so.12')
    signatures={'init':[],'load_all':[ctypes.c_char_p],'prepare':[],'run':[ctypes.c_ulonglong],'synchronize':[],'shutdown':[],'get_size':[ctypes.c_char_p,ctypes.POINTER(ctypes.c_ulonglong)]}
    for n,args in signatures.items():
        fn=getattr(lib,'srdatalog_'+n); fn.argtypes=args; fn.restype=ctypes.c_int
    def call(n,*args):
        code=getattr(lib,'srdatalog_'+n)(*args)
        if code: raise RuntimeError((n,code))
    call('init')
    try:
        call('load_all',os.fsencode(facts));call('prepare');call('synchronize')
        assert cuda.cudaProfilerStart()==0
        start=time.perf_counter();call('run',0);call('synchronize');elapsed=time.perf_counter()-start
        assert cuda.cudaProfilerStop()==0
        counts={}
        for rel in build['relations']:
            value=ctypes.c_ulonglong();call('get_size',rel['name'].encode(),ctypes.byref(value));counts[rel['name']]=value.value
        save(out,{'dataset':ds,'variant':variant,'library_sha256':build['library_sha256'],'profiled_fixedpoint_seconds':elapsed,'relation_counts':counts,'not_benchmark':True})
    finally:call('shutdown')

if __name__=='__main__':
    mode,ds=sys.argv[1:3]
    if mode=='prepare':
        from doop_suite.prepare import prepare
        old=load(Path('/data/doop-full-string-split/prepared')/ds/'manifest.json')
        new=prepare(Path('/data/doop-corrected-xlarge/raw')/ds,ROOT/'prepared'/ds,program=SOURCE/'examples/doop.py',provenance=old['provenance'])
        assert new['source_sha256']==old['source_sha256']
        assert {k:v['sha256'] for k,v in new['relations'].items()}=={k:v['sha256'] for k,v in old['relations'].items()}
        save(ROOT/'preparation'/f'{ds}.json',{'fresh':True,'all_input_relation_hashes_equal':True,'source_sha256':new['source_sha256'],'new_metadata':new['meta'],'manifest_sha256':gpu._sha256(ROOT/'prepared'/ds/'manifest.json')})
    elif mode=='build':
        variant=sys.argv[3]; out=ROOT/'builds'/ds/variant;out.mkdir(parents=True,exist_ok=True)
        if variant!='baseline':
            from candidate import program
            gpu._program=program
        gpu._build(ROOT/'prepared'/ds,out,'bitmap' if ds=='jython' else 'baseline',2)
        b=load(out/'build.json');b.update(dataset=ds,variant=variant,plan='bitmap' if ds=='jython' else 'baseline',facts=str(ROOT/'prepared'/ds),commit='07f0f19bab0ba1b8556d7871e14030ec0b3d6285')
        if variant!='baseline':b['candidate_sha256']=gpu._sha256(ROOT/'candidate.py')
        save(out/'build.json',b)
    elif mode=='run':
        variant,out=sys.argv[3:5]; b=load(ROOT/'builds'/ds/variant/'build.json')
        assert gpu._sha256(Path(b['library']))==b['library_sha256']
        gpu._execute(b,ROOT/'prepared'/ds,Path(out),Path(sys.argv[5]) if len(sys.argv)>5 else None)
    elif mode=='profile':profile(ds,sys.argv[3],Path(sys.argv[4]))
