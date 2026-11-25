import json, time, numpy as np
from pinn_try import run
from case_forward import rel_l2
import sys
mode=sys.argv[1]; res=[]; curves={}
for s in range(3):
    t=time.time(); u,l,lv=run(0.01,s,curriculum=(mode=='C')); e=rel_l2(u,0.01)
    res.append(dict(mode=mode,seed=s,err=e,loss=l,val_loss=lv,time=time.time()-t)); print(res[-1],flush=True)
    if s==0: curves[mode]=u.tolist()
json.dump(dict(res=res,curves=curves),open(f'ladder_{mode}.json','w'))
