import sys
from driver import ROOT,command
variant=sys.argv[1] if len(sys.argv)>1 else 'baseline'
suffix='' if variant=='baseline' else '-'+variant
for ds in sys.argv[2:] or ('soot','jython','scala','kotlin'):
    target=ROOT/('profiles'+suffix)/ds;target.parent.mkdir(exist_ok=True)
    command(['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-memory-usage=true','--output',str(target),sys.executable,'-B',str(ROOT/'driver.py'),'profile',ds,variant,str(target.with_suffix('.counts.json'))],ROOT/'logs'/f'{ds}{suffix}-nsys.log')
    command(['nsys','export','--type=sqlite','--output',str(target.with_suffix('.sqlite')),str(target.with_suffix('.nsys-rep'))],ROOT/'logs'/f'{ds}{suffix}-nsys-export.log')
    command([sys.executable,'-B',str(ROOT/'analyze.py'),str(target.with_suffix('.sqlite'))],ROOT/'logs'/f'{ds}{suffix}-analysis.log')
    print(ds,'profile complete',flush=True)
