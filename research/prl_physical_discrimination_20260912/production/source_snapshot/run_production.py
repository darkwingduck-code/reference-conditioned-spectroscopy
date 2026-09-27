"""Frozen production matrix for the conditional oscillator discrimination study."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from oscillator_pilot import response,rational_fit,predict
from discrimination_core import instrument_reference,measure,infer_reference_zero,spectral_decomposition
ROOT=Path(__file__).resolve().parent
Q0=.7458353261772512
CENTER=.97; HALF=.18; N=181; H=.8
OMEGA=np.linspace(CENTER-HALF,CENTER+HALF,N)
HOLD=OMEGA[:-1]+(OMEGA[1]-OMEGA[0])/2

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def groups():
 for split,qs,fields,noises,cals,reps in (
   ("primary",(Q0-.04,Q0,Q0+.04),(.025,.05,.1,.15,.2),(1e-5,1e-4,1e-3),(0.,1e-4,1e-3),40),
   ("heldout",(Q0-.02,Q0+.02),(.075,.125,.175),(1e-4,1e-3),(1e-4,),20)):
  for q in qs:
   for eta in (.003,.01):
    for noise in noises:
     for cal in cals:
      for rep in range(reps):
       yield split,q,eta,noise,cal,rep,fields

def write_csv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open("w",newline="",encoding="utf-8") as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output",default="production")
 args=parser.parse_args();out=ROOT/args.output
 if out.exists():raise FileExistsError("Preserve an existing production run; choose a new output")
 out.mkdir()
 names=("oscillator_pilot.py","discrimination_core.py","run_production.py","production_plan.md","test_discrimination.py")
 hashes={n:sha(ROOT/n) for n in names};start=time.time()
 rows=[];coveragefail=0;boxavail=0;boundfails=0;identitymax=0;sample_spectra={}
 for gi,(split,q,eta,noise,cal,rep,fields) in enumerate(groups()):
  seed_ref=2026091200+gi*101
  rng=np.random.default_rng(seed_ref)
  inst=instrument_reference(OMEGA,rng,cal)
  reftrue=response(OMEGA+1j*eta,q,0,H)
  ref,epsref,epsilon0=measure(OMEGA,reftrue,rng,noise,inst)
  boundfails+=int(np.any(abs(ref-reftrue)>epsref))
  for bi,B in enumerate(fields):
   seed_signal=seed_ref+bi+1
   rngsignal=np.random.default_rng(seed_signal)
   truth=response(OMEGA+1j*eta,q,B,H)
   chi,eps,epsilon=measure(OMEGA,truth,rngsignal,noise,inst)
   boundfails+=int(np.any(abs(chi-truth)>eps))
   inferred=infer_reference_zero(OMEGA,eta,chi,eps,ref,epsref)
   spec=spectral_decomposition(q,B,H);d=spec["full_zero"]
   row=dict(split=split,q=q,B=B,eta=eta,FWHM=2*eta,noise_radius=noise,
      calibration_radius=cal,replica=rep,seed_reference=seed_ref,seed_signal=seed_signal,
      signal_noise_absolute_radius=epsilon,reference_noise_absolute_radius=epsilon0,
      true_frequency=d,**{k:v for k,v in spec.items() if k!="full_zero"},**inferred)
   if inferred["certified"]:
    covered=inferred["frequency_lower"]<=d<=inferred["frequency_upper"]
    coveragefail+=int(not covered);boxavail+=1
    row.update(box_contains_truth=covered,reference_error=abs(inferred["inferred_frequency"]-d),
       reference_error_over_HWHM=abs(inferred["inferred_frequency"]-d)/eta,
       box_max_error_over_HWHM=inferred["maximum_frequency_error"]/eta)
   try:
    fit=rational_fit(OMEGA,chi)
    target=min(fit["zeros"],key=lambda x:abs(x-CENTER))
    heldtrue=response(HOLD+1j*eta,q,B,H)
    heldrms=float(np.linalg.norm(predict(fit,HOLD)-heldtrue)/np.linalg.norm(heldtrue))
    poles_inside=bool(np.all((fit["poles"].real>CENTER-HALF)&(fit["poles"].real<CENTER+HALF)&(fit["poles"].imag<0)))
    acceptable=bool(heldrms<=max(5*noise,1e-3) and poles_inside and fit["condition"]<1e10)
    row.update(fit_succeeded=True,fit_rms=fit["relative_rms"],holdout_clean_rms=heldrms,
       fit_condition=fit["condition"],fitted_poles_inside_lower_half_plane=poles_inside,
       fit_heuristic_acceptable=acceptable,full_fit_real=target.real,full_fit_imag=target.imag,
       full_fit_error=abs(target-(d-1j*eta)),full_fit_error_over_HWHM=abs(target-(d-1j*eta))/eta,
       pp_fit_real=fit["ppzero"].real,pp_fit_imag=fit["ppzero"].imag,
       pp_fit_error=abs(fit["ppzero"]-(d-1j*eta)),pp_fit_error_over_HWHM=abs(fit["ppzero"]-(d-1j*eta))/eta,
       pp_exact_bias=spec["pp_exact"]-d,pp_fit_minus_exact=abs(fit["ppzero"]-(spec["pp_exact"]-1j*eta)))
    if inferred["certified"]:
     row["pp_fit_outside_reference_box"]=not(inferred["frequency_lower"]<=fit["ppzero"].real<=inferred["frequency_upper"])
     row["pp_exact_outside_reference_box"]=not(inferred["frequency_lower"]<=spec["pp_exact"]<=inferred["frequency_upper"])
   except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
    row.update(fit_succeeded=False,fit_failure=str(exc))
   row["true_local_poles_inside_window"]=spec["low_pole"]>CENTER-HALF and spec["high_local_pole"]<CENTER+HALF
   rows.append(row)
   # Full finite-window inputs for an explicitly labeled illustration, not all replicas.
   if split=="primary" and abs(q-Q0)<1e-12 and eta==.003 and noise==1e-4 and cal==1e-4 and rep==0:
    sample_spectra[f"B_{B:g}_chi"]=chi;sample_spectra[f"B_{B:g}_epsilon"]=eps
    sample_spectra[f"B_{B:g}_truth"]=truth
    sample_spectra["reference"]=ref;sample_spectra["reference_epsilon"]=epsref
 if hashes!={n:sha(ROOT/n) for n in names}:raise AssertionError("Production source changed")
 write_csv(out/"trials.csv",rows)
 sample_spectra["omega"]=OMEGA
 np.savez_compressed(out/"illustration_inputs.npz",**sample_spectra)
 import pandas as pd
 df=pd.DataFrame(rows);summ=[]
 for key,g in df.groupby(["split","eta","noise_radius","calibration_radius","B"],dropna=False):
  safe=g[g.certified]
  summ.append(dict(zip(["split","eta","noise_radius","calibration_radius","B"],key))|dict(
   rows=len(g),certified=len(safe),coverage_failures=int((~safe.box_contains_truth.astype(bool)).sum()) if len(safe) else 0,
   fit_successes=int(g.fit_succeeded.sum()),heuristic_acceptable=int(g.fit_heuristic_acceptable.fillna(False).sum()),
   pp_outside_box=int(safe.pp_fit_outside_reference_box.fillna(False).sum()),
   max_reference_error_over_HWHM=float(safe.reference_error_over_HWHM.max()) if len(safe) else None,
   max_box_error_over_HWHM=float(safe.box_max_error_over_HWHM.max()) if len(safe) else None,
   median_pp_error_over_HWHM=float(g.pp_fit_error_over_HWHM.median()),
   median_full_fit_error_over_HWHM=float(g.full_fit_error_over_HWHM.median()),
   median_holdout_rms=float(g.holdout_clean_rms.median())))
 write_csv(out/"summary.csv",summ)
 report={"status":"COMPLETED" if not(coveragefail or boundfails) else "FAILED",
  "scope":"Synthetic passive finite oscillator; conditional star-model parameter boxes, no material or universal continuation certificate",
  "rows":len(rows),"primary_rows":int((df.split=="primary").sum()),"heldout_rows":int((df.split=="heldout").sum()),
  "conditional_boxes":boxavail,"conditional_box_coverage_failures":coveragefail,"pointwise_calibration_bound_failures":boundfails,
  "fit_failures":int((~df.fit_succeeded).sum()),"seconds":time.time()-start,"source_sha256":hashes,
  "all_source_bytes_unchanged":True,"noise_distribution":"Independent complex uniform disk with declared radius; calibration and baseline shared across fields within one q/eta/replica group",
  "holdout_scope":"New q/B combinations are held out from the pilot; interlaced frequencies score against synthetic noiseless truth only after fitting",
  "file_sha256":{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
 (out/"diagnostic.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()

