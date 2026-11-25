"""Case study 1, mitigated PINN: hard boundary conditions (trial function x(1-x)N(x), as in
Lagaris et al.) and collocation points graded towards the boundary layer at x = 1."""
import json, time
import numpy as np
import torch
from case_forward import XE, rel_l2

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)


def pinn_fixed(eps, width, seed, n_col=256, adam_steps=4000, lbfgs_steps=500):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, 1))
    s = np.linspace(0, 1, n_col)
    xg = 1 - (1 - s) ** 3          # graded: dense near x = 1
    x = torch.tensor(xg).reshape(-1, 1).requires_grad_(True)
    u_of = lambda z: z * (1 - z) * net(z)

    def loss_fn():
        u = u_of(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        return ((-eps * d2u + du - 1.0) ** 2).mean()

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(adam_steps):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
    opt2 = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt2.zero_grad(); l = loss_fn(); l.backward(); return l
    opt2.step(closure)
    with torch.no_grad():
        z = torch.tensor(XE).reshape(-1, 1)
        u = u_of(z).numpy().ravel()
    return u, float(loss_fn().detach())


if __name__ == "__main__":
    res, curves = [], {}
    for eps in (0.1, 0.01):
        for s in range(3):
            t0 = time.perf_counter(); u, l = pinn_fixed(eps, 32, s); t = time.perf_counter() - t0
            e = rel_l2(u, eps); res.append(dict(eps=eps, seed=s, err=e, loss=l, time=t))
            print(eps, s, e, l, round(t, 1), flush=True)
            if s == 0:
                curves[f"pinnfix_{eps}"] = u.tolist()
    json.dump(res, open("forward_pinnfix.json", "w"), indent=1)
    json.dump(curves, open("forward_pinnfix_curves.json", "w"))
