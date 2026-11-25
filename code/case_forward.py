"""Case study 1 (forward problem): steady convection-diffusion with a boundary layer.

    -eps u'' + u' = 1 on (0,1),  u(0) = u(1) = 0,
    exact: u(x) = x - (exp((x-1)/eps) - exp(-1/eps)) / (1 - exp(-1/eps)).

Compared under one protocol (same exact solution, same error metric, same
evaluation grid, several seeds, wall time recorded):
  * P1 Galerkin FEM and P1 SUPG FEM (classical, certified a priori theory)
  * ELM / random-feature collocation (random tanh features + least squares)
  * PINN (tanh MLP, Adam then L-BFGS)
"""
import json, time, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
XE = np.linspace(0, 1, 4001)


def exact(x, eps):
    return x - (np.exp((x - 1) / eps) - np.exp(-1 / eps)) / (1 - np.exp(-1 / eps))


def rel_l2(u, eps):
    ue = exact(XE, eps)
    return float(np.sqrt(np.trapezoid((u - ue) ** 2, XE) / np.trapezoid(ue ** 2, XE)))


# ---------------------------------------------------------------- FEM
def fem(eps, n_el, supg=False):
    h = 1.0 / n_el
    n = n_el - 1  # interior nodes
    diff = eps / h * np.array([-1, 2, -1])
    conv = 0.5 * np.array([-1, 0, 1])
    A = np.zeros((n, n)); b = np.full(n, h)
    tau = 0.0
    if supg:
        pe = h / (2 * eps)
        tau = h / 2 * (1 / np.tanh(pe) - 1 / pe)
        # streamline diffusion adds tau*(u',v') and rhs tau*(1, v') (=0 for interior hats)
        diff = (eps + tau) / h * np.array([-1, 2, -1])
    for i in range(n):
        for k, j in enumerate((i - 1, i, i + 1)):
            if 0 <= j < n:
                A[i, j] += diff[k] + conv[k]
    u = np.zeros(n_el + 1)
    u[1:-1] = np.linalg.solve(A, b)
    xn = np.linspace(0, 1, n_el + 1)
    return np.interp(XE, xn, u)


# ---------------------------------------------------------------- ELM
def elm(eps, m, seed, n_col=None):
    rng = np.random.default_rng(seed)
    n_col = n_col or 2 * m
    # random features; slopes scaled so that some features resolve the layer
    w = rng.uniform(-1, 1, m) * (1.0 / np.sqrt(eps)) * 2.0
    c = rng.uniform(0, 1, m)
    bias = -w * c
    xc = 0.5 * (1 - np.cos(np.pi * np.arange(n_col) / (n_col - 1)))  # Chebyshev-like points
    xi = xc[1:-1]
    t = np.tanh(np.outer(xi, w) + bias)
    d1 = (1 - t ** 2) * w
    d2 = -2 * t * (1 - t ** 2) * w ** 2
    H = np.vstack([-eps * d2 + d1, np.tanh(np.outer([0.0, 1.0], w) + bias) * 10.0])
    T = np.concatenate([np.ones(len(xi)), [0.0, 0.0]])
    beta, *_ = np.linalg.lstsq(H, T, rcond=1e-13)
    return np.tanh(np.outer(XE, w) + bias) @ beta, float(np.linalg.cond(H))


# ---------------------------------------------------------------- PINN
def pinn(eps, width, seed, n_col=256, adam_steps=4000, lbfgs_steps=500):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, width), torch.nn.Tanh(),
                              torch.nn.Linear(width, 1))
    x = torch.linspace(0, 1, n_col).reshape(-1, 1).requires_grad_(True)
    xb = torch.tensor([[0.0], [1.0]])

    def loss_fn():
        u = net(x)
        du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
        r = -eps * d2u + du - 1.0
        return (r ** 2).mean() + 10.0 * (net(xb) ** 2).mean()

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(adam_steps):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
    opt2 = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt2.zero_grad(); l = loss_fn(); l.backward(); return l
    opt2.step(closure)
    with torch.no_grad():
        u = net(torch.tensor(XE).reshape(-1, 1)).numpy().ravel()
    nparams = sum(p.numel() for p in net.parameters())
    return u, float(loss_fn().detach()), nparams


if __name__ == "__main__":
    out = {"fem": [], "supg": [], "elm": [], "pinn": []}
    curves = {}
    for eps in (0.1, 0.01):
        for n_el in (8, 16, 32, 64, 128, 256, 512):
            t0 = time.perf_counter(); u = fem(eps, n_el); t = time.perf_counter() - t0
            out["fem"].append(dict(eps=eps, dof=n_el - 1, err=rel_l2(u, eps), time=t))
            t0 = time.perf_counter(); u = fem(eps, n_el, supg=True); t = time.perf_counter() - t0
            out["supg"].append(dict(eps=eps, dof=n_el - 1, err=rel_l2(u, eps), time=t))
        for m in (16, 32, 64, 128, 256):
            errs, times, conds = [], [], []
            for s in range(5):
                t0 = time.perf_counter(); u, cnd = elm(eps, m, s); times.append(time.perf_counter() - t0)
                errs.append(rel_l2(u, eps)); conds.append(cnd)
            out["elm"].append(dict(eps=eps, dof=m, err=errs, time=times, cond=conds))
            print("elm", eps, m, np.median(errs), flush=True)
        for width in (16, 32):
            for s in range(3):
                t0 = time.perf_counter(); u, l, npar = pinn(eps, width, s); t = time.perf_counter() - t0
                e = rel_l2(u, eps)
                out["pinn"].append(dict(eps=eps, width=width, seed=s, dof=npar, err=e, loss=l, time=t))
                print("pinn", eps, width, s, e, l, round(t, 1), flush=True)
                if width == 32 and s == 0:
                    curves[f"pinn_{eps}"] = u.tolist()
        curves[f"fem32_{eps}"] = fem(eps, 32).tolist()
        curves[f"supg32_{eps}"] = fem(eps, 32, supg=True).tolist()
        curves[f"elm64_{eps}"] = elm(eps, 64, 0)[0].tolist()
        curves[f"exact_{eps}"] = exact(XE, eps).tolist()
    json.dump(out, open("forward_results.json", "w"), indent=1)
    json.dump(curves, open("forward_curves.json", "w"))
