from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lattice_kubo import *
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data_prl'/'full_bz_completion';mu0=.7;T=.06;rows=[];wr=[]
for L in [6,8]:
 p=TorusParams(Lx=L,Ly=L,Nkz=16);q=2*np.pi/p.Nkz;omega=2j*q;s={n:diagonalize_torus(p,n) for n in [-1,0,1]};n0=torus_density(s[0],mu0,T);mus={0:mu0,-1:torus_mu(s[-1],n0,T),1:torus_mu(s[1],n0,T)}
 for n in [-1,0,1]:
  for ens,m in [('fixed_mu',mu0),('fixed_n',mus[n])]:
   c=torus_kubo(s[n],m,T,1,omega);rows.append(dict(L=L,nphi=n,B=s[n].B,ensemble=ens,mu=m,density=torus_density(s[n],m,T),total_re=c.total.real,total_im=c.total.imag,conduction_re=c.conduction.real,conduction_im=c.conduction.imag,interband_re=c.interband.real,interband_im=c.interband.imag,other_re=c.other.real,other_im=c.other.imag))
  if n:
   w=torus_ward(s[n],mus[n],T,1,2*q+.02j)
   for V in [0.,.8]:
    K=w.K if V==0 else rpa_dress(w.K,V);Q=2*np.sin(q/2);qv=np.array([2*q+.02j,-Q]);wr.append(dict(L=L,nphi=n,B=s[n].B,V=V,vertex=w.vertex,left=np.max(abs(qv@K)),right=np.max(abs(K@qv))))
r=pd.DataFrame(rows);r.to_csv(OUT/'magnetic_torus_full_band.csv',index=False);pd.DataFrame(wr).to_csv(OUT/'magnetic_torus_ward_rpa.csv',index=False)
coef=[]
for L in [6,8]:
 B=2*np.pi/L**2
 for ens in ['fixed_mu','fixed_n']:
  x=r[(r.L==L)&(r.ensemble==ens)].set_index('nphi')
  for c in ['total','conduction','interband','other']:
   zm=complex(x.loc[-1,c+'_re'],x.loc[-1,c+'_im']);z0=complex(x.loc[0,c+'_re'],x.loc[0,c+'_im']);zp=complex(x.loc[1,c+'_re'],x.loc[1,c+'_im']);v=(.5*(zm+zp)-z0)/B**2;coef.append(dict(L=L,B=B,ensemble=ens,component=c,c2_re=v.real,c2_im=v.imag))
pd.DataFrame(coef).to_csv(OUT/'magnetic_torus_even_coefficients.csv',index=False)
summary=dict(max_vertex=float(pd.DataFrame(wr).vertex.max()),max_left=float(pd.DataFrame(wr).left.max()),max_right=float(pd.DataFrame(wr).right.max()))
(OUT/'magnetic_torus_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
