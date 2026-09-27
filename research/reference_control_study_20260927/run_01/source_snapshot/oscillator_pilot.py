"""Passive three-oscillator pilot. Existing research inputs are read-only."""
from __future__ import annotations
import hashlib,json,csv,time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent

def stiffness(q,B,h=.8, dark_curvature=0., dark_remote=0.):
    # Squared-frequency stiffness; q, B, h are dimensionless model controls.
    wb2=1.+(.3*q)**2
    wd=1.3*q+dark_curvature*B*B
    return np.array([[wb2,.8*B,h],[.8*B,wd*wd,dark_remote],[h,dark_remote,2.6**2]])

def modes(q,B,h=.8,**kw):
    values,vectors=np.linalg.eigh(stiffness(q,B,h,**kw))
    if min(values)<=0: raise ValueError("Unstable stiffness")
    return np.sqrt(values),vectors

def response(z,q,B,h=.8,source=None,**kw):
    z=np.asarray(z,complex)
    source=np.array([1.,0.,0.]) if source is None else np.asarray(source,complex)
    w,V=modes(q,B,h,**kw)
    weights=np.abs(V.conj().T@source)**2
    return np.sum(weights/(z[...,None]**2-w*w),axis=-1)

def direct(z,q,B,h=.8,source=None,**kw):
    source=np.array([1.,0.,0.]) if source is None else np.asarray(source,complex)
    return source.conj()@np.linalg.solve(z*z*np.eye(3)-stiffness(q,B,h,**kw),source)

def true_pp_zero(q,B,h=.8):
    w,V=modes(q,B,h)
    R=np.abs(V[0,:2])**2/(2*w[:2])
    return (R[0]*w[1]+R[1]*w[0])/sum(R)

def rational_fit(z,y,numerator_degree=3,denominator_degree=2):
    z=np.asarray(z,complex); y=np.asarray(y,complex)
    center=np.mean(z); scale=max(abs(z-center))
    t=(z-center)/scale
    # monic denominator, centered powers; equation error initialization
    n=numerator_degree; d=denominator_degree
    matrix=np.column_stack([y*t**k for k in range(d-1,-1,-1)]+[-t**k for k in range(n,-1,-1)])
    x,_,rank,_=np.linalg.lstsq(matrix,-y*t**d,rcond=None)
    den=np.r_[1.,x[:d]]; num=x[d:]
    quotient,rem=np.polydiv(num,den)
    poles=center+scale*np.roots(den)
    zeros=center+scale*np.roots(num)
    pp=center-scale*rem[-1]/rem[-2] if len(rem)==2 and abs(rem[-2])>1e-20 else complex(np.nan)
    prediction=np.polyval(num,t)/np.polyval(den,t)
    return dict(center=center,scale=scale,num=num,den=den,poles=poles,zeros=zeros,ppzero=pp,
                relative_rms=float(np.linalg.norm(prediction-y)/np.linalg.norm(y)),condition=float(np.linalg.cond(matrix)))

def predict(fit,z):
    t=(np.asarray(z)-fit["center"])/fit["scale"]
    return np.polyval(fit["num"],t)/np.polyval(fit["den"],t)

def crossing(h=.8):
    return brentq(lambda q:(1.+(.3*q)**2-(1.3*q)**2)*(2.6**2-(1.3*q)**2)-h*h,.5,1.)

def main():
    out=ROOT/"pilot";out.mkdir(exist_ok=True)
    rows=[]
    for h in (0.,.4,.8,1.2):
      q0=crossing(h)
      for q in (q0-.08,q0,q0+.08):
       for B in (.025,.05,.1,.15,.2):
        for eta in (.003,.01):
         for halfwidth in (.10,.18,.28):
          center=1.3*q
          omega=np.linspace(center-halfwidth,center+halfwidth,241)
          z=omega+1j*eta
          y=response(z,q,B,h)
          fit=rational_fit(omega,y)
          full=min(fit["zeros"],key=lambda t:abs(t-center))
          hold=omega[:-1]+(omega[1]-omega[0])/2
          held=response(hold+1j*eta,q,B,h)
          holdrms=float(np.linalg.norm(predict(fit,hold)-held)/np.linalg.norm(held))
          rows.append(dict(h=h,q=q,q0=q0,B=B,eta=eta,halfwidth=halfwidth,
             true_zero=center,pp_exact=true_pp_zero(q,B,h),pp_fit_real=fit["ppzero"].real,
             pp_fit_imag=fit["ppzero"].imag,full_fit_real=full.real,full_fit_imag=full.imag,
             pp_bias=true_pp_zero(q,B,h)-center,full_error=abs(full-(center-1j*eta)),
             fit_rms=fit["relative_rms"],holdout_rms=holdrms,condition=fit["condition"]))
    with (out/"pilot.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    candidates=[r for r in rows if r["holdout_rms"]<1e-4]
    scored=sorted(candidates,key=lambda r:abs(r["pp_bias"])/max(r["full_error"],1e-15),reverse=True)
    summary={"scope":"pilot, not production or class-flip evidence","cases":len(rows),"holdout_below_1e-4":len(candidates),
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"best_examples":scored[:8],
             "q0_by_h":{str(h):crossing(h) for h in (0.,.4,.8,1.2)}}
    (out/"pilot.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()

