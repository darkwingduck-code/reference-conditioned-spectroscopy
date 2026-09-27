from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent))
from landau_projection import *
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'data_prl'/'full_bz_completion';rng=np.random.default_rng(260624)
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.array([[1,0],[0,-1]],complex);pauli=[SX,SY,SZ]
def fib(n):
 i=np.arange(n);z=1-2*(i+.5)/n;ph=np.pi*(3-np.sqrt(5))*i;r=np.sqrt(1-z*z);return np.c_[r*np.cos(ph),r*np.sin(ph),z]
def d(k):x,y,z=k;return np.array([np.sin(x),np.sin(y),2-np.cos(x)-np.cos(y)-np.cos(z)])
def vel(k):
 x,y,z=k;v=d(k);E=np.linalg.norm(v);return np.array([v@np.array([np.cos(x),0,np.sin(x)])/E,v@np.array([0,np.cos(y),np.sin(y)])/E,v@np.array([0,0,np.sin(z)])/E])
mu=.35;dirs=fib(24);st=[];ar=[];vv=[];zz=[];nn=[];vvec=[]
for chi,z0 in [(1,np.pi/2),(-1,-np.pi/2)]:
 k0=np.array([0.,0.,z0])
 for nh in dirs:
  f=lambda r:np.linalg.norm(d(k0+r*nh))-mu;hi=.2
  while f(hi)<0:hi*=1.5
  rad=brentq(f,0,hi);k=k0+rad*nh;dv=d(k);_,u=np.linalg.eigh(dv[0]*SX+dv[1]*SY+dv[2]*SZ);v=vel(k);area=rad**2*np.linalg.norm(v)/abs(nh@v)*(4*np.pi/len(dirs))
  st.append(u[:,-1]);ar.append(area);vv.append(np.linalg.norm(v));zz.append(.72+.06*chi*nh[2]+.03*nh[0]);nn.append(chi);vvec.append(v)
fs=FermiSurface(np.array(st),np.array(ar),np.array(vv),np.array(zz),np.array(nn));P=fs.n;I=np.eye(2);dens=np.einsum('ac,bd->abcd',I,I);spin=sum(np.einsum('ac,bd->abcd',s,s) for s in pauli);G=np.zeros((P,P,2,2,2,2),complex);vvec=np.array(vvec);nn=np.array(nn)
for i in range(P):
 for j in range(P):
  ang=vvec[i]@vvec[j]/np.linalg.norm(vvec[i])/np.linalg.norm(vvec[j]);g=(1.1 if nn[i]==nn[j] else .45)+.25*ang+.1*nn[i]*nn[j];G[i,j]=150*(g*dens+.12*spin)
f=project_gamma(fs,G);diag=mode_diagnostics(fs,f,{1:1.,-1:-1.});perr=[];eerr=[]
for _ in range(30):
 ph=rng.uniform(-np.pi,np.pi,P);fp=project_gamma(apply_phases(fs,ph),G);dp=mode_diagnostics(apply_phases(fs,ph),fp,{1:1.,-1:-1.});perr.append(np.max(abs(fp-f)));eerr.append(np.max(abs(dp['eigenvalues']-diag['eigenvalues'])))
pd.DataFrame(dict(mode=np.arange(P),eigenvalue=diag['eigenvalues'],charge_overlap=diag['charge_overlap'],axial_overlap=diag['axial_overlap'])).to_csv(OUT/'exact_landau_spectrum.csv',index=False)
# non-Abelian U(2) covariance
P2,A,R=4,5,2;U=np.empty((P2,A,R),complex);Z=np.empty((P2,R,R),complex);W=np.empty((P2,R,R),complex);N=np.empty((P2,R,R),complex)
def unitary(n):
 z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));q,r=np.linalg.qr(z);p=np.diag(r);p=np.where(abs(p)>0,p/abs(p),1);return q@np.diag(p.conj())
for i in range(P2):
 U[i],_=np.linalg.qr(rng.normal(size=(A,R))+1j*rng.normal(size=(A,R)));u=unitary(R);Z[i]=u@np.diag([.5,.8])@u.conj().T;W[i]=unitary(R);a=rng.normal(size=(R,R))+1j*rng.normal(size=(R,R));N[i]=.5*(a+a.conj().T)
GG=rng.normal(size=(P2,P2,A,A,A,A))+1j*rng.normal(size=(P2,P2,A,A,A,A));GG=.5*(GG+GG.transpose(1,0,3,2,5,4).conj());ff=project_subspaces(U,Z,GG);U2,Z2=rotate_subspaces(U,Z,W);ff2=project_subspaces(U2,Z2,GG);pred=rotate_projected(ff,W);N2=rotate_density(N,W);de=abs(subspace_energy(ff,N)-subspace_energy(ff2,N2))
summary=dict(model='lattice-derived Fermi-surface patches and dynamic forward vertex',patches=P,U1_projection_residual=max(perr),U1_eigenvalue_residual=max(eerr),relative_two_mode_leakage=diag['relative_leakage'],two_mode_matrix_re=diag['two_mode'].real.tolist(),leading_eigenvalues=diag['eigenvalues'][:8].tolist(),leading_charge_overlap=diag['charge_overlap'][:8].tolist(),leading_axial_overlap=diag['axial_overlap'][:8].tolist(),nonabelian_covariance_residual=float(np.max(abs(ff2-pred))),nonabelian_energy_residual=float(de))
(OUT/'exact_landau_projection_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
