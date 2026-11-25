import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,"axes.spines.top":False,"axes.spines.right":False,"legend.frameon":False,"axes.linewidth":0.6})
d=json.load(open("operator_results.json"))
names=list(d["tests"].keys()); short=["training\n($\\ell$=0.1,\n$\\sigma$=1)","rougher\n($\\ell$=0.05)","contrast\n($\\sigma$=2)","both"]
cols=["#2CA02C","#1F77B4","#FF7F0E","#D62728"]
fig,ax=plt.subplots(1,2,figsize=(5.3,2.5),gridspec_kw=dict(wspace=0.38,width_ratios=[1,1.1]))
a=ax[0]
for i,(n,c) in enumerate(zip(names,cols)):
    e=np.array(d["tests"][n]["err"]); jit=np.random.default_rng(i).uniform(-0.18,0.18,len(e))
    a.scatter(i+jit,e,s=3,color=c,alpha=0.45,lw=0)
    a.plot([i-0.28,i+0.28],[np.median(e)]*2,color="k",lw=1.2)
a.set_yscale("log"); a.set_xticks(range(4)); a.set_xticklabels(short,fontsize=5.8); a.set_ylabel("relative $L^2$ error"); a.set_title("(a) error by test distribution",fontsize=7.5)
b=ax[1]
thr=np.percentile(d["tests"][names[0]]["res"],95)
for n,c,s in zip(names,cols,short):
    b.scatter(d["tests"][n]["res"],d["tests"][n]["err"],s=3,color=c,alpha=0.5,lw=0,label=s.replace("\n"," "))
b.axvline(thr,color="0.3",ls="--",lw=0.7); b.axhline(0.05,color="0.3",ls=":",lw=0.7)
b.text(thr*1.15,0.5,"flag\nthreshold",fontsize=5.8,color="0.3",va="center")
b.text(0.55,0.058,"5% error",fontsize=5.8,color="0.3")
b.set_xscale("log"); b.set_yscale("log"); b.set_xlabel("per-query residual of the prediction"); b.set_ylabel("relative $L^2$ error")
b.set_title("(b) residual as an error indicator",fontsize=7.5); b.legend(fontsize=5.6,loc="lower right",markerscale=2.5,handletextpad=0.2)
fig.savefig("../fig/fig_case3.pdf",bbox_inches="tight"); fig.savefig("../fig/fig_case3.png",dpi=160,bbox_inches="tight")
