"""Test of the small-D preference in case I: PINN runs started from large D and small r."""
import json, time, sys
import numpy as np
from case_inverse import pinn, data
inits=[(0.15,0.0),(0.12,0.1),(0.10,0.2)]
out=[]
y=data(1)
for k,(D0,r0) in enumerate(inits):
    t=time.time(); rr=pinn(1,y,D0,r0,seed=10+k); rr['time']=time.time()-t; out.append(rr); print(rr,flush=True)
    json.dump(out,open('inverse_case1_bias.json','w'),indent=1)
