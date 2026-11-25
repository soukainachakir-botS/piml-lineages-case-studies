"""Case study 2 (inverse problem): identifiability of (D, r) in

    u_t = D u_xx - r u  on (0,1) x (0,T],  u(0,t) = u(1,t) = 0,

from noisy point sensors. Two initial conditions:
  case I : u0 = sin(pi x)                      -> data depend on D*pi^2 + r only
  case II: u0 = sin(pi x) + 0.5 sin(3 pi x)    -> two decay rates, (D, r) identifiable

Each case is solved twice from several random initial guesses:
  (a) classical output least squares with the exact forward model,
  (b) PINN joint optimisation over (theta, D, r).
The Fisher information matrix and Cramer-Rao bound are computed analytically.
"""
import json, sys, time
import numpy as np
from scipy.optimize import least_squares
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
PI = np.pi
D_TRUE, R_TRUE, SIG = 0.05, 0.5, 0.01
XS = np.array([0.2, 0.5, 0.7])
TS = np.linspace(0, 1, 21)
XX, TT = np.meshgrid(XS, TS, indexing="ij")
XX, TT = XX.ravel(), TT.ravel()


def model(D, r, x, t, case):
    u = np.exp(-(D * PI**2 + r) * t) * np.sin(PI * x)
    if case == 2:
        u = u + 0.5 * np.exp(-(9 * D * PI**2 + r) * t) * np.sin(3 * PI * x)
    return u


def sens(D, r, x, t, case):
    e1 = np.exp(-(D * PI**2 + r) * t) * np.sin(PI * x)
    S = np.stack([-t * PI**2 * e1, -t * e1], 1)
    if case == 2:
        e3 = 0.5 * np.exp(-(9 * D * PI**2 + r) * t) * np.sin(3 * PI * x)
        S += np.stack([-t * 9 * PI**2 * e3, -t * e3], 1)
    return S


def data(case, seed=123):
    rng = np.random.default_rng(seed)
    return model(D_TRUE, R_TRUE, XX, TT, case) + SIG * rng.standard_normal(XX.size)


def classical(case, y, inits):
    res = []
    for D0, r0 in inits:
        sol = least_squares(lambda p: (model(p[0], p[1], XX, TT, case) - y) / SIG, [D0, r0],
                            jac=lambda p: sens(p[0], p[1], XX, TT, case) / SIG,
                            bounds=([0.0, 0.0], [0.5, 5.0]), method="trf", x_scale=[0.05, 0.5])
        res.append(dict(D=float(sol.x[0]), r=float(sol.x[1]), cost=float(2 * sol.cost / y.size)))
    return res


def pinn(case, y, D0, r0, seed, adam=5000, lbfgs=800):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32),
                              torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 1))
    D = torch.tensor(D0, requires_grad=True); r = torch.tensor(r0, requires_grad=True)
    rng = np.random.default_rng(seed)
    xc = torch.tensor(rng.uniform(0, 1, (2000, 2)), requires_grad=True)  # (x,t)
    xi = torch.linspace(0, 1, 101)
    ic = torch.stack([xi, torch.zeros_like(xi)], 1)
    u0 = torch.sin(PI * xi) + (0.5 * torch.sin(3 * PI * xi) if case == 2 else 0)
    tb = torch.linspace(0, 1, 51)
    bc = torch.cat([torch.stack([torch.zeros_like(tb), tb], 1), torch.stack([torch.ones_like(tb), tb], 1)])
    xd = torch.tensor(np.stack([XX, TT], 1)); yd = torch.tensor(y)

    def loss_fn():
        u = net(xc)
        g = torch.autograd.grad(u.sum(), xc, create_graph=True)[0]
        uxx = torch.autograd.grad(g[:, 0].sum(), xc, create_graph=True)[0][:, 0]
        res = g[:, 1] - D * uxx + r * u[:, 0]
        l_pde = (res ** 2).mean()
        l_ic = ((net(ic)[:, 0] - u0) ** 2).mean()
        l_bc = (net(bc) ** 2).mean()
        l_d = (((net(xd)[:, 0] - yd) / SIG) ** 2).mean() * SIG**2 * 100  # noise-weighted, rescaled
        return l_pde + l_ic + l_bc + l_d, l_d

    params = list(net.parameters()) + [D, r]
    opt = torch.optim.Adam(params, lr=2e-3)
    for _ in range(adam):
        opt.zero_grad(); l, _ = loss_fn(); l.backward(); opt.step()
    opt2 = torch.optim.LBFGS(params, max_iter=lbfgs, line_search_fn="strong_wolfe",
                             tolerance_grad=1e-12, tolerance_change=1e-15)

    def closure():
        opt2.zero_grad(); l, _ = loss_fn(); l.backward(); return l
    opt2.step(closure)
    l, ld = loss_fn()
    return dict(D=float(D.detach()), r=float(r.detach()), loss=float(l.detach()), D0=D0, r0=r0)


if __name__ == "__main__":
    case = int(sys.argv[1]); n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rng = np.random.default_rng(7)
    inits = [(float(a), float(b)) for a, b in zip(rng.uniform(0.01, 0.15, 12), rng.uniform(0.0, 1.5, 12))]
    y = data(case)
    out = dict(case=case, D_true=D_TRUE, r_true=R_TRUE, sigma=SIG)
    S = sens(D_TRUE, R_TRUE, XX, TT, case)
    F = S.T @ S / SIG**2
    ev = np.linalg.eigvalsh(F)
    out["fisher"] = F.tolist(); out["fisher_eigs"] = ev.tolist()
    out["fisher_cond"] = float(ev[-1] / max(ev[0], 1e-300))
    if ev[0] > 1e-8 * ev[-1]:
        C = np.linalg.inv(F); out["crb_std"] = np.sqrt(np.diag(C)).tolist(); out["crb_cov"] = C.tolist()
    out["classical"] = classical(case, y, inits)
    out["pinn"] = []
    for k in range(n_runs):
        t0 = time.perf_counter()
        rr = pinn(case, y, inits[k][0], inits[k][1], seed=k)
        rr["time"] = time.perf_counter() - t0
        out["pinn"].append(rr); print(case, k, rr, flush=True)
        json.dump(out, open(f"inverse_case{case}.json", "w"), indent=1)
