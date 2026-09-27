from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lattice_kubo import *
from ensemble import hall_compensated
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'data_prl'/'full_bz_completion';OUT.mkdir(parents=True,exist_ok=True)
mu0=.7;T=.05;Bs=np.array([.0008,.0012,.0016,.0020]);p=SlabParams();q=2*np.pi/p.Nkz;omega=2j*q
s0=diagonalize_slab(p,0);n0=density(s0,mu0,T);c0=density_kubo(s0,mu0,T,1,omega)
rows=[];wards=[]
for B in Bs:
 sm=diagonalize_slab(p,-B);sp=diagonalize_slab(p,B);mu=common_mu_pair(sm,sp,n0,T)
 ne=.5*(density(sm,mu0,T)+density(sp,mu0,T));rows.append(dict(B=B,quantity='density',ensemble='fixed_mu',component='total',q2=(ne-n0)/B**2,mu2q=(mu-mu0)/B**2))
 for ens,m in [('fixed_mu',mu0),('fixed_n',mu)]:
  cm=density_kubo(sm,m,T,1,omega);cp=density_kubo(sp,m,T,1,omega)
  for comp in ['total','conduction','interband','other']:
   z0=getattr(c0,comp);z=.5*(getattr(cm,comp)+getattr(cp,comp));d=(z-z0)/B**2
   rows.append(dict(B=B,quantity='response',ensemble=ens,component=comp,q2=d.real,q2_imag=d.imag,mu2q=(mu-mu0)/B**2))
 wr=ward_tensor(sp,mu,T,1,2*q+.015j)
 for V in [0.,.7,1.4]:
  K=wr.K if V==0 else rpa_dress(wr.K,V);Q=2*np.sin(q/2);qv=np.array([2*q+.015j,-Q])
  wards.append(dict(B=B,V=V,vertex=wr.vertex,left=np.max(abs(qv@K)),right=np.max(abs(K@qv)),contact_re=wr.contact.real))
r=pd.DataFrame(rows);r.to_csv(OUT/'slab_full_band_raw.csv',index=False);pd.DataFrame(wards).to_csv(OUT/'slab_ward_rpa.csv',index=False)
def fit(sub,col='q2',imag=False):
 x=sub.B.to_numpy()**2;y=sub[col].to_numpy();return np.polyfit(x,y,2)[-1]
dens=r[r.quantity=='density'];n2=fit(dens);mu2=fit(dens,'mu2q');fits=[]
for ens in ['fixed_mu','fixed_n']:
 for comp in ['total','conduction','interband','other']:
  s=r[(r.quantity=='response')&(r.ensemble==ens)&(r.component==comp)];fits.append(dict(ensemble=ens,component=comp,c2_re=fit(s),c2_im=fit(s,'q2_imag')))
f=pd.DataFrame(fits);f.to_csv(OUT/'slab_full_band_fits.csv',index=False)
h=2e-5;n0p=(density(s0,mu0+h,T)-density(s0,mu0-h,T))/(2*h);K0p=(density_kubo(s0,mu0+h,T,1,omega).total-density_kubo(s0,mu0-h,T,1,omega).total)/(2*h)
kmu=f[(f.ensemble=='fixed_mu')&(f.component=='total')].iloc[0];kn=f[(f.ensemble=='fixed_n')&(f.component=='total')].iloc[0];K2mu=complex(kmu.c2_re,kmu.c2_im);K2n=complex(kn.c2_re,kn.c2_im);mu2c,K2c=hall_compensated(n0p=n0p,n2=n2,K0p=K0p,K2=K2mu)
summary=dict(model='two-copy Hall-compensated lattice Weyl slab (B-even implementation)',Lx=p.Lx,Nky=p.Nky,Nkz=p.Nkz,mu0=mu0,T=T,q=q,n0=n0,n0p=n0p,n2=n2,mu2_fit=mu2,mu2_chain=mu2c,mu2_relerr=abs(mu2/mu2c-1),K2mu_re=K2mu.real,K2mu_im=K2mu.imag,K2n_re=K2n.real,K2n_im=K2n.imag,K2chain_re=complex(K2c).real,K2chain_im=complex(K2c).imag,K2_relerr=abs((K2n-K2c)/K2n),max_vertex=float(pd.DataFrame(wards).vertex.max()),max_left=float(pd.DataFrame(wards).left.max()),max_right=float(pd.DataFrame(wards).right.max()))
(OUT/'slab_chain_rule_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
