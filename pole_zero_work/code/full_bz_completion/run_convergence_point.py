"""Run one (Lx,Nky,B) convergence point; aggregate points externally."""
from pathlib import Path
import argparse,sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from lattice_kubo import *
p0=argparse.ArgumentParser();p0.add_argument('--Lx',type=int,required=True);p0.add_argument('--Nky',type=int,required=True);p0.add_argument('--B',type=float,required=True);p0.add_argument('--output',required=True);a=p0.parse_args()
p=SlabParams(Lx=a.Lx,Nky=a.Nky);mu0=.7;T=.05;q=2*np.pi/p.Nkz;w=2j*q;s0=diagonalize_slab(p,0);sm=diagonalize_slab(p,-a.B);sp=diagonalize_slab(p,a.B);n0=density(s0,mu0,T);mu=common_mu_pair(sm,sp,n0,T);c0=density_kubo(s0,mu0,T,1,w).total
cm=density_kubo(sm,mu0,T,1,w).total;cp=density_kubo(sp,mu0,T,1,w).total;nm=density_kubo(sm,mu,T,1,w).total;np_=density_kubo(sp,mu,T,1,w).total;B=a.B
out=dict(Lx=a.Lx,Nky=a.Nky,B=B,n2_quotient=(.5*(density(sm,mu0,T)+density(sp,mu0,T))-n0)/B**2,mu2_quotient=(mu-mu0)/B**2,K2_mu_quotient_re=((.5*(cm+cp)-c0)/B**2).real,K2_n_quotient_re=((.5*(nm+np_)-c0)/B**2).real)
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
