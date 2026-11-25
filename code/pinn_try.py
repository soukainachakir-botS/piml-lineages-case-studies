import numpy as np, torch, time, sys
from case_forward import XE, rel_l2
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
def run(eps, seed, curriculum=True, n_u=256, n_g=256, steps=3000, lb=500):
    torch.manual_seed(seed)
    net=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,1))
    s=np.linspace(0,1,n_g); xg=np.concatenate([np.linspace(0,1,n_u),1-(1-s)**3])
    x=torch.tensor(np.sort(xg)).reshape(-1,1).requires_grad_(True)
    u_of=lambda z: z*(1-z)*net(z)
    def loss(e, xx=x):
        u=u_of(xx); du=torch.autograd.grad(u.sum(),xx,create_graph=True)[0]; d2=torch.autograd.grad(du.sum(),xx,create_graph=True)[0]
        return ((-e*d2+du-1)**2).mean()
    sched=[0.1,0.05,0.02,eps] if curriculum else [eps]
    for e in sched:
        opt=torch.optim.Adam(net.parameters(),lr=1e-3)
        for _ in range(steps):
            opt.zero_grad(); l=loss(e); l.backward(); opt.step()
        o2=torch.optim.LBFGS(net.parameters(),max_iter=lb,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
        def cl():
            o2.zero_grad(); l=loss(e); l.backward(); return l
        o2.step(cl)
    xv=torch.tensor(np.linspace(0,1,5001)).reshape(-1,1).requires_grad_(True)
    with torch.no_grad(): u=u_of(torch.tensor(XE).reshape(-1,1)).numpy().ravel()
    return u, float(loss(eps).detach()), float(loss(eps,xv).detach())
if __name__=='__main__':
    t=time.time(); u,l,lv=run(0.01,int(sys.argv[1]),curriculum=sys.argv[2]=='1'); print(sys.argv[1:],rel_l2(u,0.01),l,lv,time.time()-t)
