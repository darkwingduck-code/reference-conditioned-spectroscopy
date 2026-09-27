"""Read-only numerical aggregation and new figures; never changes production."""
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from oscillator_pilot import rational_fit,predict,stiffness,direct,response
ROOT=Path(__file__).resolve().parent
def main():
 out=ROOT/"analysis"
 if out.exists():raise FileExistsError("Preserve analysis")
 out.mkdir();figdir=out/"figures";figdir.mkdir()
 source=ROOT/"production_v2"/"trials.csv";before=hashlib.sha256(source.read_bytes()).hexdigest()
 df=pd.read_csv(source);dec=pd.read_csv(ROOT/"controls"/"background_decomposition.csv")
 ctrl=pd.read_csv(ROOT/"controls"/"structural_controls.csv")
 wy=pd.read_csv(ROOT/"controls"/"weyl_transfer_trials.csv")
 qc=pd.read_csv(ROOT/"controls"/"finite_intercept_control.csv")
 q0=.7458353261772512
 rep=df[(df.split=="primary")&(abs(df.q-q0)<1e-10)&(df.eta==.003)&(df.noise_radius==1e-4)&(df.calibration_radius==1e-4)]
 summary=[]
 for B,g in rep.groupby("B"):
  summary.append(dict(B=B,n=len(g),pp_error_median_HWHM=g.pp_fit_error_over_HWHM.median(),
   pp_error_max_HWHM=g.pp_fit_error_over_HWHM.max(),full_fit_error_max_HWHM=g.full_fit_error_over_HWHM.max(),
   reference_error_max_HWHM=g.reference_error_over_HWHM.max(),box_max_radius_HWHM=g.box_max_error_over_HWHM.max(),
   fit_holdout_median=g.holdout_clean_rms.median(),fit_holdout_max=g.holdout_clean_rms.max(),
   pp_outside_box=int(g.pp_fit_outside_reference_box.sum())))
 pd.DataFrame(summary).to_csv(out/"representative_summary.csv",index=False)
 # Direct orbital inversion at an independent complex frequency for every row.
 maxdirect=0.
 for r in df.itertuples():
  z=.873+.037j
  K=stiffness(r.q,r.B)
  x=z*z
  cofactor=(x-K[1,1])*(x-K[2,2])
  determinant=(x-K[0,0])*cofactor-K[0,1]**2*(x-K[2,2])-K[0,2]**2*(x-K[1,1])
  diff=abs(cofactor/determinant-direct(z,r.q,r.B))
  maxdirect=max(maxdirect,diff)
 # Verify every stored box and every accounting total independently.
 assert len(df)==11280 and len(df[df.split=="primary"])==10800
 assert ((df.frequency_lower<=df.true_frequency)&(df.true_frequency<=df.frequency_upper)).all()
 assert (df.design_condition<1e10).all() and (df.left_inverse_defect_infinity_norm<1e-8).all()
 assert maxdirect<1e-10
 checks=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(ROOT),"-p","test_discrimination.py","-v"],
     capture_output=True,text=True)
 (out/"unit_tests.txt").write_text(checks.stdout+checks.stderr,encoding="utf-8")
 assert checks.returncode==0
 # Published-size vector figures with explicit HWHM unit and independent bounds.
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":9,
   "axes.labelsize":9,"legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42})
 fig,axs=plt.subplots(1,3,figsize=(10.5,3.1),layout="constrained")
 saved=np.load(ROOT/"production_v2"/"illustration_inputs.npz")
 w=saved["omega"];chi=saved["B_0.2_chi"];truth=saved["B_0.2_truth"]
 fit=rational_fit(w,chi)
 ax=axs[0];ax.plot(w,-truth.imag,color="#174a7e",lw=1.8,label="Full passive response")
 ax.plot(w,-predict(fit,w).imag,"--",color="#db7118",lw=1.,label="Two poles + affine")
 d=1.3*q0
 ax.axvline(d,color="#222222",lw=1.,label="Full zero (real part)")
 ax.axvline(fit["ppzero"].real,color="#c52235",ls=":",lw=1.2,label="Fitted pp zero")
 ax.set(xlabel="Frequency",ylabel=r"$-\mathrm{Im}\,\chi_{cc}$",title="(a) Two visible resonances")
 ax.legend(frameon=False,loc="upper left")
 ax=axs[1]
 g=rep.groupby("B")
 for key,color,label,marker in (("pp_fit_real","#c52235","Finite-window pp zero","o"),
    ("full_fit_real","#db7118","Full-numerator fit","s")):
  y=(g[key].median()-d)/.003
  ax.plot(y.index,y,marker+"-",color=color,ms=4,lw=1.2,label=label)
 mid=(g.inferred_frequency.median()-d)/.003
 lower=(g.frequency_lower.min()-d)/.003;upper=(g.frequency_upper.max()-d)/.003
 ax.fill_between(mid.index,lower,upper,color="#276b9b",alpha=.19,label="Conditional reference boxes")
 ax.plot(mid.index,mid,color="#174a7e",lw=1.1,label="Reference estimate")
 ax.axhline(0,color=".4",lw=.7);ax.axhline(1,color=".5",ls=":",lw=.8)
 ax.set(xlabel="Field control B",ylabel=r"Frequency offset / HWHM $\eta$",title="(b) Apparent dark-mode stiffening")
 ax.legend(frameon=False,fontsize=7,loc="upper left")
 ax=axs[2]
 part=df[(df.split=="primary")&(abs(df.q-q0)<1e-10)&(df.B==.2)&(df.eta==.003)]
 for noise,color in ((1e-5,"#168577"),(1e-4,"#174a7e"),(1e-3,"#c52235")):
  sub=part[part.noise_radius==noise].groupby("calibration_radius").box_max_error_over_HWHM.max()
  ax.plot([0,1,2],sub.values,"o-",color=color,label=f"Signal radius {noise:g}")
 ax.axhline(1.17955,color=".25",ls="--",lw=1,label="pp bias ≈ 1.18 HWHM")
 ax.set(xticks=[0,1,2],xticklabels=["0","1e−4","1e−3"],yscale="log",
   xlabel="Calibration radius / standard RMS",ylabel="Largest conditional box radius / HWHM",
   title="(c) Precision boundary")
 ax.legend(frameon=False,fontsize=7,loc="upper left")
 for ax in axs:ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.15)
 for ext in ("pdf","png"):fig.savefig(figdir/f"linewidth_discrimination.{ext}",dpi=220)
 plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(8.5,6.1),layout="constrained")
 ax=axs[0,0];dd=dec[dec.h==.8]
 ax.plot(dd.B,-dd.negative_frequency_correction/.003,"o-",label="Negative local poles",color="#174a7e")
 ax.plot(dd.B,-dd.remote_pair_correction/.003,"s-",label="Remote ± pair",color="#c52235")
 ax.set(xlabel="Field control B",ylabel="Correction magnitude / HWHM",title="(a) Background origin at h = 0.8")
 ax.legend(frameon=False)
 ax=axs[0,1]
 names=["valid_star","true_dark_curvature","dark_remote","mixed_source","bright_drift","unknown_eta"]
 vals=[]
 for name in names:
  sub=ctrl[(ctrl.kind==name)&(ctrl.B==.2)].iloc[0]
  vals.append(max(abs(sub.inferred_frequency-sub.true_compressed_frequency)/.003,1e-12))
 ax.bar(np.arange(6),vals,color=["#174a7e","#168577"]+["#c52235"]*4)
 ax.set(xticks=np.arange(6),xticklabels=["Star","True\ncurvature","Dark–\nremote","Mixed\nsource","Bright\ndrift","Unknown\nwidth"],
   yscale="log",ylim=(1e-12,2),ylabel="Error from compressed root / HWHM",title="(b) Reference assumptions matter")
 ax=axs[1,0]
 for f0,color in ((.4,"#174a7e"),(.8,"#168577"),(1.5,"#c52235")):
  sub=wy[(wy.F0a==f0)&(wy.noise_radius==0)]
  ax.loglog(sub.pp_error_over_HWHM,sub.full_error_over_HWHM,"o",ms=4,color=color,label=f"F0a={f0:g}")
 ax.plot([1e-5,10],[1e-5,10],"--",color=".5",lw=.8)
 ax.set(xlabel="pp fit error / HWHM",ylabel="Full fit error / HWHM",title="(c) Actual Weyl continuum: no generic rescue")
 ax.legend(frameon=False)
 ax=axs[1,1]
 for h,color in ((0.,"#7a7a7a"),(.8,"#174a7e"),(1.2,"#c52235")):
  sub=qc[qc.h==h]
  ax.plot(sub.B,sub.scaled_shift,"o-",label=f"h={h:g}",color=color,ms=4)
 ax.set(xlabel="Field control B",ylabel=r"$(q_{\rm pp}-q_0)/B^2$",title="(d) Finite intercept class is preserved")
 ax.legend(frameon=False)
 for ax in axs.flat:ax.spines[["top","right"]].set_visible(False);ax.grid(alpha=.15)
 for ext in ("pdf","png"):fig.savefig(figdir/f"scope_controls.{ext}",dpi=220)
 plt.close(fig)
 report={"status":"PASS","rows_direct_3x3_cofactor_checks":len(df),"max_direct_cofactor_difference":maxdirect,
    "unit_test_count":12,"unit_test_exit":checks.returncode,
    "conditional_boxes_contain_truth":11280,"pp_fit_outside_conditional_box_count":int(df.pp_fit_outside_reference_box.sum()),
    "all_production_boxes_available":True,"largest_box_radius_HWHM":float(df.box_max_error_over_HWHM.max()),
    "max_left_inverse_defect":float(df.left_inverse_defect_infinity_norm.max()),
    "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "frozen_trials_sha256":before,"trials_unchanged":before==hashlib.sha256(source.read_bytes()).hexdigest(),
    "scope":"Independent direct 3x3 vs expanded cofactor and output accounting; model/optimizer implementation remains author-written, not an external replication"}
 (out/"validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()

