import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from case_forward import XE, exact

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
C = dict(exact="#222222", fem="#1F77B4", supg="#8C564B", elm="#2CA02C", pinn="#D62728", pinnfix="#FF7F0E")

R = json.load(open("forward_results.json"))
CV = json.load(open("forward_curves.json"))
LB, LC = json.load(open("ladder_B.json")), json.load(open("ladder_C.json"))
FX = json.load(open("forward_pinnfix.json"))

fig = plt.figure(figsize=(5.3, 5.0)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.38)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
# (a) solutions near the layer, eps = 0.01
a = ax[0]; m = XE > 0.8
a.plot(XE[m], np.array(CV["exact_0.01"])[m], color=C["exact"], lw=2.2, alpha=0.35, label="exact")
a.plot(XE[m], np.array(CV["fem32_0.01"])[m], color=C["fem"], lw=1, label="Galerkin FEM, 32 el.")
a.plot(XE[m], np.array(CV["supg32_0.01"])[m], color=C["supg"], lw=1, ls="--", label="SUPG FEM, 32 el.")
a.plot(XE[m], np.array(CV["elm64_0.01"])[m], color=C["elm"], lw=1, ls="-.", label="ELM, 64 features")
a.plot(XE[m], np.array(CV["pinn_0.01"])[m], color=C["pinn"], lw=1, label="PINN, plain")
a.plot(XE[m], np.array(LC["curves"]["C"])[m], color=C["pinnfix"], lw=1, ls=":", label="PINN, hard BC\n+ continuation")
a.set_xlabel("$x$"); a.set_ylabel("$u(x)$"); a.set_title(r"(a) solutions near the layer, $\varepsilon=0.01$", fontsize=7.5)
a.set_ylim(-0.95, 1.3); a.legend(fontsize=5.8, loc="lower left", handlelength=1.8, ncol=2, columnspacing=0.6)

# (b) error versus number of unknowns
b = ax[1]
for eps, mk, ls in ((0.1, "o", "-"), (0.01, "s", "--")):
    f = [r for r in R["fem"] if r["eps"] == eps]
    b.loglog([r["dof"] for r in f], [r["err"] for r in f], ls, marker=mk, ms=2.5, color=C["fem"], lw=0.9)
    e = [r for r in R["elm"] if r["eps"] == eps]
    d = np.array([r["dof"] for r in e]); med = np.array([np.median(r["err"]) for r in e])
    lo = np.array([min(r["err"]) for r in e]); hi = np.array([max(r["err"]) for r in e])
    b.errorbar(d, med, yerr=[med - lo, hi - med], fmt=mk, ls=ls, ms=2.5, color=C["elm"], lw=0.9, capsize=1.5)
    p = [r for r in R["pinn"] if r["eps"] == eps]
    for npar in sorted({r["dof"] for r in p}):
        errs = [r["err"] for r in p if r["dof"] == npar]
        b.errorbar([npar], [np.median(errs)], yerr=[[np.median(errs) - min(errs)], [max(errs) - np.median(errs)]],
                   fmt=mk, ms=3, color=C["pinn"], capsize=1.5, lw=0.9)
xs = np.array([10, 500]); b.loglog(xs, 2.5 * xs ** -2.0, color="0.6", lw=0.6)
b.text(9, 2e-5, "slope $-2$", color="0.45", fontsize=6)
b.set_xlabel("unknowns (FEM, ELM) or parameters (PINN)"); b.set_ylabel(r"relative $L^2$ error")
b.set_title("(b) accuracy versus size", fontsize=7.5); b.set_ylim(1e-13, 3)
from matplotlib.lines import Line2D
h = [Line2D([], [], color=C["fem"], marker="o", ms=3, lw=0.9, label="Galerkin FEM"),
     Line2D([], [], color=C["elm"], marker="o", ms=3, lw=0.9, label="ELM (median, min--max)"),
     Line2D([], [], color=C["pinn"], marker="o", ms=3, lw=0, label="PINN, plain"),
     Line2D([], [], color="k", marker="o", ms=3, lw=0.9, label=r"$\varepsilon=0.1$"),
     Line2D([], [], color="k", marker="s", ms=3, lw=0.9, ls="--", label=r"$\varepsilon=0.01$")]
b.legend(handles=h, fontsize=6.2, loc="lower left")

# (c) PINN mitigation ladder at eps = 0.01
c = ax[2]
plain = [r for r in R["pinn"] if r["eps"] == 0.01 and r["width"] == 32]
A = [r for r in FX if r["eps"] == 0.01]
rows = [("plain", [r["err"] for r in plain], [r["loss"] for r in plain], None),
        ("hard BC\ngraded", [r["err"] for r in A], [r["loss"] for r in A], [1329.8]),
        ("hard BC\nmixed", [r["err"] for r in LB["res"]], [r["loss"] for r in LB["res"]], [r["val_loss"] for r in LB["res"]]),
        ("+ continuation", [r["err"] for r in LC["res"]], [r["loss"] for r in LC["res"]], [r["val_loss"] for r in LC["res"]])]
for i, (lab, err, loss, val) in enumerate(rows):
    c.scatter([i - 0.12] * len(err), err, s=10, color=C["pinn"], zorder=3, label="relative $L^2$ error" if i == 0 else None)
    c.scatter([i + 0.12] * len(loss), loss, s=10, marker="^", color="0.35", zorder=3, label="training residual" if i == 0 else None)
    if val:
        c.scatter([i + 0.12] * len(val), val, s=14, marker="x", color=C["pinnfix"], zorder=3,
                  label="residual on fresh points" if i == 1 else None)
c.set_yscale("log"); c.set_xticks(range(4)); c.set_xticklabels([r[0].replace("\n"," ") for r in rows], fontsize=7)
c.set_title(r"(c) PINN remedies, $\varepsilon=0.01$, 3 seeds", fontsize=7.5); c.set_ylim(1e-5, 1e4)
c.legend(fontsize=6.2, loc="upper right")
fig.savefig("../fig/fig_case1.pdf", bbox_inches="tight"); fig.savefig("../fig/fig_case1.png", dpi=150, bbox_inches="tight")

# ------------------------------------------------------------------ inverse
fig = plt.figure(figsize=(5.3, 5.0)); gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
PI2 = np.pi ** 2
for k, case in enumerate((1, 2)):
    d = json.load(open(f"inverse_case{case}.json")); a = ax[k]
    Dg = np.linspace(-0.03, 0.16, 50)
    a.plot(Dg, 1.0 * (0.05 * PI2 + 0.5) - PI2 * Dg, color="0.55", lw=0.8, ls="--", label=r"$D\pi^2+r=$ const.")
    cl = np.array([[r["D"], r["r"]] for r in d["classical"]])
    pn = np.array([[r["D"], r["r"]] for r in d["pinn"]])
    p0 = np.array([[r["D0"], r["r0"]] for r in d["pinn"]])
    a.scatter(cl[:, 0], cl[:, 1], marker="x", s=16, color=C["fem"], lw=0.9, label="least squares, exact model", zorder=3)
    a.scatter(p0[:, 0], p0[:, 1], s=10, facecolor="none", edgecolor="0.6", lw=0.6, label="initial guesses (PINN)")
    for s0, s1 in zip(p0, pn):
        a.annotate("", xy=s1, xytext=s0, arrowprops=dict(arrowstyle="-|>", lw=0.4, color="0.7", mutation_scale=5))
    a.scatter(pn[:, 0], pn[:, 1], s=14, color=C["pinn"], label="PINN estimates", zorder=4)
    a.scatter([0.05], [0.5], marker="*", s=60, color="k", zorder=5, label="true value")
    if "crb_cov" in d:
        Cv = np.array(d["crb_cov"]); w, V = np.linalg.eigh(Cv)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        a.add_patch(Ellipse((0.05, 0.5), 2 * 2.447 * np.sqrt(w[1]), 2 * 2.447 * np.sqrt(w[0]), angle=ang,
                            fill=False, color="k", lw=0.7, label="95% Cramér–Rao region"))
    a.set_xlim(-0.03, 0.16); a.set_ylim(-0.1, 2.35); a.set_xlabel("$D$"); a.set_ylabel("$r$")
    a.set_title("(a) case I: $u_0=\\sin\\pi x$" if case == 1 else "(b) case II: $u_0=\\sin\\pi x+\\frac{1}{2}\\sin 3\\pi x$",
                fontsize=7.5)
    if case == 1:
        a.legend(fontsize=5.8, loc="upper right", ncol=1, borderaxespad=0.1)
    else:
        a.legend(handles=[a.patches[-1]], fontsize=6, loc="upper right")
        ins = a.inset_axes([0.55, 0.12, 0.42, 0.38])
        ins.scatter(pn[:, 0], pn[:, 1], s=8, color=C["pinn"]); ins.scatter(cl[:1, 0], cl[:1, 1], marker="x", s=12, color=C["fem"])
        ins.scatter([0.05], [0.5], marker="*", s=30, color="k")
        ins.add_patch(Ellipse((0.05, 0.5), 2 * 2.447 * np.sqrt(w[1]), 2 * 2.447 * np.sqrt(w[0]), angle=ang, fill=False, color="k", lw=0.6))
        ins.set_xlim(0.0475, 0.0525); ins.set_ylim(0.47, 0.535); ins.tick_params(labelsize=4.5); ins.set_xticks([0.048, 0.052]); ins.set_yticks([0.48, 0.52])
# (c) losses
a = ax[2]
for case, mk in ((1, "o"), (2, "s")):
    d = json.load(open(f"inverse_case{case}.json"))
    pn = np.array([[r["D"], r["loss"]] for r in d["pinn"]])
    a.scatter(np.abs(pn[:, 0] - 0.05) / 0.05, pn[:, 1], marker=mk, s=14, color=C["pinn"] if case == 1 else C["elm"],
              label=f"case {'I' if case == 1 else 'II'}")
a.set_xscale("log"); a.set_xlabel(r"relative error in $D$"); a.set_ylabel("final PINN training loss")
a.set_title("(c) loss does not reveal it", fontsize=7.5); a.legend(fontsize=6); a.set_ylim(5.5e-3, 8e-3)
fig.savefig("../fig/fig_case2.pdf", bbox_inches="tight"); fig.savefig("../fig/fig_case2.png", dpi=150, bbox_inches="tight")
print("ok")
