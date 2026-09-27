"""Independent control matrix and a reduced-Weyl finite-window transfer check."""
from pathlib import Path
import hashlib,json,csv,sys
import numpy as np
from scipy.optimize import brentq
from oscillator_pilot import response,direct,stiffness,rational_fit,predict,crossing,true_pp_zero
from discrimination_core import infer_reference_zero,compressed_zeros,spectral_decomposition
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/"pole_zero_work"/"code"))
from weyl_continuum_stress import crossing_model,isolated_roots,node_response
def write_csv(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 out=ROOT/"controls"
 if out.exists():raise FileExistsError("Preserve controls")
 out.mkdir()
 omega=np.linspace(.79,1.15,181);eta=.003;q=.7458353261772512
 rows=[];decomp=[];crosses=[]
 for h in (0.,.4,.8,1.2):
  q0=crossing(h)
  for B in (.025,.05,.1,.15,.2):
   decomp.append(dict(h=h,B=B,q=q0,**spectral_decomposition(q0,B,h)))
   qp=brentq(lambda qq:true_pp_zero(qq,B,h)-np.sqrt(np.linalg.eigvalsh(stiffness(qq,0,h)[np.ix_([0,2],[0,2])])[0]),q0-.1,q0+.1)
   crosses.append(dict(h=h,B=B,true_q0=q0,pp_crossing_q=qp,pp_crossing_shift=qp-q0,scaled_shift=(qp-q0)/(B*B)))
 for B in (.05,.1,.2):
  for kind in ("valid_star","true_dark_curvature","dark_remote","mixed_source","bright_drift","unknown_eta"):
   kw={};source=None;shifteta=eta
   if kind=="true_dark_curvature":kw["dark_curvature"]=.08
   if kind=="dark_remote":kw["dark_remote"]=.2*B
   if kind=="mixed_source":source=[1,0,.3]
   if kind=="unknown_eta":shifteta=eta+.05*B*B
   K=stiffness(q,B,**kw)
   if kind=="bright_drift":K[0,0]+=.2*B*B
   v=np.array([1,0,0]) if source is None else np.array(source)
   chi=np.array([v@np.linalg.solve((w+1j*shifteta)**2*np.eye(3)-K,v) for w in omega])
   ref=response(omega+1j*eta,q,0,source=source)
   inf=infer_reference_zero(omega,eta,chi,np.full(181,1e-9),ref,np.full(181,1e-9))
   compressed=compressed_zeros(q,B,source=source,**kw)[0]
   actualbare=np.sqrt(K[1,1])
   rows.append(dict(kind=kind,B=B,true_bare_frequency=actualbare,true_compressed_frequency=float(compressed),
     structural_assumptions_hold=kind in ("valid_star","true_dark_curvature"),
     **inf,reported_box_contains_compressed=inf.get("frequency_lower",-np.inf)<=compressed<=inf.get("frequency_upper",np.inf),
     reported_box_contains_bare=inf.get("frequency_lower",-np.inf)<=actualbare<=inf.get("frequency_upper",np.inf)))
 write_csv(out/"structural_controls.csv",rows);write_csv(out/"background_decomposition.csv",decomp)
 write_csv(out/"finite_intercept_control.csv",crosses)
 # This second model does not satisfy the oscillator reference identity.
 # It is a transfer/limitation check, not an extension of the conditional box.
 rng=np.random.default_rng(2026091288);wr=[]
 upstream=("weyl_continuum_stress.py","weyl_fl_prb_solver.py","pole_zero_tomography.py")
 before={n:hashlib.sha256((REPO/"pole_zero_work"/"code"/n).read_bytes()).hexdigest() for n in upstream}
 for f0 in (.4,.8,1.5):
  model,qw,center=crossing_model(f0);edge=center-1.
  for B in (.003,.006,.012):
   poles,zero=isolated_roots(model,B)
   for gr in (.05,.2):
    gamma=gr*edge
    for hw in (.2,.5,1.):
     x=np.linspace(center-hw*edge,center+hw*edge,201)
     y=np.array([node_response(model,w+1j*gamma,B) for w in x])
     hold=x[:-1]+(x[1]-x[0])/2
     hy=np.array([node_response(model,w+1j*gamma,B) for w in hold])
     for noise in (0.,1e-4):
      for rep in range(1 if noise==0 else 10):
       eps=noise*np.linalg.norm(y)/np.sqrt(len(y))
       yn=y+eps*np.sqrt(rng.uniform(size=len(y)))*np.exp(2j*np.pi*rng.uniform(size=len(y)))
       fit=rational_fit(x,yn)
       chosen=min(fit["zeros"],key=lambda z:abs(z-center))
       pp=fit["ppzero"];target=zero-1j*gamma
       hres=float(np.linalg.norm(predict(fit,hold)-hy)/np.linalg.norm(hy))
       wr.append(dict(F0a=f0,q=qw,B=B,gamma_over_edge=gr,halfwidth_over_edge=hw,noise_radius=noise,replica=rep,
         true_full_zero_real=zero,true_full_zero_imag=-gamma,pp_fit_real=pp.real,pp_fit_imag=pp.imag,
         full_fit_real=chosen.real,full_fit_imag=chosen.imag,full_error_over_HWHM=abs(chosen-target)/gamma,
         pp_error_over_HWHM=abs(pp-target)/gamma,fit_rms=fit["relative_rms"],holdout_rms=hres,
         chosen_inside_window=bool(x[0]<chosen.real<x[-1] and chosen.imag<=0),
         certificate_available=False))
 write_csv(out/"weyl_transfer_trials.csv",wr)
 assert before=={n:hashlib.sha256((REPO/"pole_zero_work"/"code"/n).read_bytes()).hexdigest() for n in upstream}
 valid=[r for r in rows if r["structural_assumptions_hold"]]
 report=dict(status="COMPLETED",structural_rows=len(rows),decomposition_rows=len(decomp),intercept_rows=len(crosses),
    valid_star_box_failures=sum(not r["reported_box_contains_bare"] for r in valid),
    weyl_rows=len(wr),weyl_noiseless_rows=sum(r["noise_radius"]==0 for r in wr),
    weyl_scope="Actual reduced Weyl logarithmic response, uniform broadening, local rational fit, no oscillator certificate and no collision theory",
    upstream_source_sha256=before,upstream_unchanged=True,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/"diagnostic.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()

