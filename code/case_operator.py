"""Case study 3 (operator learning): 1D Darcy flow -(a(x) u')' = 1 on (0,1), u(0)=u(1)=0.

The operator a -> u is learned by a Fourier neural operator (FNO) from N_train pairs with
log a drawn from a Gaussian random field of length scale l_train. It is then tested
  * in distribution  (same length scale and contrast),
  * out of distribution: rougher coefficients (smaller length scale), higher contrast.
For every test input we record the relative L2 error and a per-query physics residual of
the prediction, computed with the same finite-volume operator used to generate the data.
"""
import json, time, sys
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
torch.set_num_threads(2)
N = 128                      # grid points (cell centres)
x = (np.arange(N) + 0.5) / N
h = 1.0 / N


def grf(n, ell, sigma, rng):
    """log-coefficient samples: Gaussian process with squared-exponential kernel."""
    d = x[:, None] - x[None, :]
    C = sigma ** 2 * np.exp(-0.5 * (d / ell) ** 2) + 1e-8 * np.eye(N)
    L = np.linalg.cholesky(C)
    return (L @ rng.standard_normal((N, n))).T


def fv_matrix(a):
    """finite-volume matrix for -(a u')' with homogeneous Dirichlet BCs at x=0,1."""
    af = np.empty(N + 1)
    af[1:-1] = 2 * a[1:] * a[:-1] / (a[1:] + a[:-1])     # harmonic mean on interior faces
    af[0], af[-1] = a[0], a[-1]
    A = np.zeros((N, N))
    for i in range(N):
        wl = af[i] / (h * h) * (2 if i == 0 else 1)
        wr = af[i + 1] / (h * h) * (2 if i == N - 1 else 1)
        A[i, i] = wl + wr
        if i > 0: A[i, i - 1] = -wl
        if i < N - 1: A[i, i + 1] = -wr
    return A


def solve(a):
    return np.linalg.solve(fv_matrix(a), np.ones(N))


def residual(a, u):
    """relative discrete residual ||A(a)u - 1|| / ||1|| of a predicted field."""
    return float(np.linalg.norm(fv_matrix(a) @ u - 1.0) / np.sqrt(N))


def dataset(n, ell, sigma, seed):
    rng = np.random.default_rng(seed)
    loga = grf(n, ell, sigma, rng)
    a = np.exp(loga)
    u = np.stack([solve(ai) for ai in a])
    return a.astype(np.float32), u.astype(np.float32)


class SpectralConv1d(torch.nn.Module):
    def __init__(self, c, modes):
        super().__init__()
        self.modes = modes
        self.w = torch.nn.Parameter((1 / c) * torch.randn(c, c, modes, dtype=torch.cfloat))

    def forward(self, v):
        vh = torch.fft.rfft(v)
        out = torch.zeros(v.shape[0], v.shape[1], v.shape[-1] // 2 + 1, dtype=torch.cfloat)
        out[:, :, :self.modes] = torch.einsum("bik,iok->bok", vh[:, :, :self.modes], self.w)
        return torch.fft.irfft(out, n=v.shape[-1])


class FNO1d(torch.nn.Module):
    def __init__(self, modes=16, width=48, layers=4, pad=32):
        super().__init__()
        self.pad = pad
        self.P = torch.nn.Linear(2, width)
        self.K = torch.nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.W = torch.nn.ModuleList([torch.nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.Q = torch.nn.Sequential(torch.nn.Linear(width, 128), torch.nn.GELU(), torch.nn.Linear(128, 1))

    def forward(self, a):
        g = torch.tensor(x, dtype=torch.float32).expand(a.shape[0], N)
        v = self.P(torch.stack([torch.log(a), g], -1)).permute(0, 2, 1)
        v = torch.nn.functional.pad(v, (0, self.pad))          # padding for non-periodic data
        for i, (k, w) in enumerate(zip(self.K, self.W)):
            v = k(v) + w(v)
            if i < len(self.K) - 1:
                v = torch.nn.functional.gelu(v)
        v = v[..., :N].permute(0, 2, 1)
        return self.Q(v).squeeze(-1)


def rel(u, v):
    return np.linalg.norm(u - v, axis=1) / np.linalg.norm(v, axis=1)


if __name__ == "__main__":
    t0 = time.time()
    a_tr, u_tr = dataset(1000, 0.10, 1.0, 1)
    sc = float(u_tr.std())
    tests = {"in-distribution": dataset(200, 0.10, 1.0, 2),
             "rougher (l=0.05)": dataset(200, 0.05, 1.0, 3),
             "higher contrast (sigma=2)": dataset(200, 0.10, 2.0, 4),
             "rougher and higher contrast": dataset(200, 0.05, 2.0, 5)}
    print("data", time.time() - t0, flush=True)
    torch.manual_seed(0)
    net = FNO1d()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    A, U = torch.tensor(a_tr), torch.tensor(u_tr) / sc
    for ep in range(epochs):
        perm = torch.randperm(len(A))
        for i in range(0, len(A), 50):
            idx = perm[i:i + 50]
            p = net(A[idx])
            loss = (torch.linalg.norm(p - U[idx], dim=1) / torch.linalg.norm(U[idx], dim=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0: print(ep, float(loss), flush=True)
    train_time = time.time() - t0
    out = {"train_time": train_time, "n_params": sum(p.numel() for p in net.parameters()), "tests": {}}
    for name, (a, u) in tests.items():
        t1 = time.time()
        with torch.no_grad():
            p = net(torch.tensor(a)).numpy() * sc
        dt = (time.time() - t1) / len(a)
        err = rel(p, u)
        res = np.array([residual(ai, pi) for ai, pi in zip(a, p)])
        out["tests"][name] = dict(err=err.tolist(), res=res.tolist(), time_per_query=dt)
        print(name, np.median(err), np.percentile(err, 95), np.median(res), flush=True)
    t1 = time.time(); [solve(ai) for ai in tests["in-distribution"][0][:50]]
    out["fv_time_per_solve"] = (time.time() - t1) / 50
    json.dump(out, open("operator_results.json", "w"))
