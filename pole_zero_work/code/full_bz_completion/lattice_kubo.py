"""UV-complete two-band lattice Weyl benchmarks with Peierls coupling.

The slab form permits a smooth B->0 derivative.  The magnetic-torus form
uses quantized flux and has no physical surfaces.  In both cases the density,
link current, and contact vertex are differentiated from the same lattice
Hamiltonian, so the finite-difference Ward identity is exact.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq

SX=np.array([[0,1],[1,0]],complex)
SY=np.array([[0,-1j],[1j,0]],complex)
SZ=np.array([[1,0],[0,-1]],complex)

@dataclass(frozen=True)
class SlabParams:
    Lx:int=22; Nky:int=64; Nkz:int=36
    m:float=2.; tx:float=1.; ty:float=1.; tz:float=1.; vx:float=1.; vy:float=1.
    theta_y:float=.371; theta_z:float=.217
    @property
    def dim(self): return 2*self.Lx
    @property
    def volume(self): return self.Lx*self.Nky*self.Nkz
    def ky_grid(self): return (2*np.pi*np.arange(self.Nky)+self.theta_y)/self.Nky-np.pi
    def kz_grid(self): return (2*np.pi*np.arange(self.Nkz)+self.theta_z)/self.Nkz-np.pi

@dataclass
class Spectrum:
    p:SlabParams; B:float; ky:np.ndarray; kz:np.ndarray; evals:np.ndarray; evecs:np.ndarray

def slab_hamiltonian(p:SlabParams,ky:float,kz:float,B:float):
    H=np.zeros((p.dim,p.dim),complex); Tx=-.5*p.tx*SZ-.5j*p.vx*SX; x0=.5*(p.Lx-1)
    for x in range(p.Lx):
        kyt=ky+B*(x-x0); sl=slice(2*x,2*x+2)
        H[sl,sl]=(p.m-p.ty*np.cos(kyt)-p.tz*np.cos(kz))*SZ+p.vy*np.sin(kyt)*SY
        if x+1<p.Lx:
            sp=slice(2*(x+1),2*(x+1)+2); H[sp,sl]=Tx; H[sl,sp]=Tx.conj().T
    return H

def diagonalize_slab(p:SlabParams,B:float):
    ky=p.ky_grid(); kz=p.kz_grid(); E=np.empty((p.Nky,p.Nkz,p.dim)); U=np.empty((p.Nky,p.Nkz,p.dim,p.dim),complex)
    for iy,y in enumerate(ky):
        for iz,z in enumerate(kz): E[iy,iz],U[iy,iz]=sla.eigh(slab_hamiltonian(p,float(y),float(z),float(B)),check_finite=False)
    return Spectrum(p,float(B),ky,kz,E,U)

def fermi(e,mu,T):
    x=(np.asarray(e)-mu)/T; out=np.empty_like(x,float); pos=x>=0
    a=np.exp(-np.minimum(x[pos],700)); out[pos]=a/(1+a)
    a=np.exp(np.maximum(x[~pos],-700)); out[~pos]=1/(1+a)
    return out

def density(spec:Spectrum,mu:float,T:float): return float(np.sum(fermi(spec.evals,mu,T))/spec.p.volume)

def common_mu_pair(sm:Spectrum,sp:Spectrum,target:float,T:float):
    lo=min(sm.evals.min(),sp.evals.min())-1; hi=max(sm.evals.max(),sp.evals.max())+1
    return float(brentq(lambda m:.5*(density(sm,m,T)+density(sp,m,T))-target,lo,hi,xtol=1e-13,rtol=1e-13))

@dataclass(frozen=True)
class Components:
    total:complex; conduction:complex; interband:complex; other:complex

def density_kubo(spec:Spectrum,mu:float,T:float,qshift:int,omega:complex):
    p=spec.p; sh=qshift%p.Nkz; occ=fermi(spec.evals,mu,T); total=intra=inter=other=0j
    for iy in range(p.Nky):
        for iz in range(p.Nkz):
            j=(iz+sh)%p.Nkz; en=spec.evals[iy,iz]; em=spec.evals[iy,j]
            R=spec.evecs[iy,iz].conj().T@spec.evecs[iy,j]
            df=occ[iy,iz][:,None]-occ[iy,j][None,:]
            t=df*np.abs(R)**2/(complex(omega)+en[:,None]-em[None,:])
            total+=t.sum(); ep=en[:,None]>0; mp=em[None,:]>0; mi=ep&mp; mx=ep^mp
            intra+=t[mi].sum(); inter+=t[mx].sum(); other+=t[~(mi|mx)].sum()
    v=p.volume; return Components(total/v,intra/v,inter/v,other/v)

@dataclass(frozen=True)
class WardResult:
    vertex:float; left:float; right:float; contact:complex; K:np.ndarray

def ward_tensor(spec:Spectrum,mu:float,T:float,qshift:int,omega:complex):
    p=spec.p; sh=qshift%p.Nkz; q=2*np.pi*sh/p.Nkz; Q=2*np.sin(q/2); z=complex(omega); occ=fermi(spec.evals,mu,T)
    K00=Kz0=K0z=Kzz=cl=cr=0j; vr=0.
    for iy,y in enumerate(spec.ky):
        for iz,k in enumerate(spec.kz):
            j=(iz+sh)%p.Nkz; kp=spec.kz[j]; en=spec.evals[iy,iz]; em=spec.evals[iy,j]
            U=spec.evecs[iy,iz]; V=spec.evecs[iy,j]
            G=(slab_hamiltonian(p,float(y),float(kp),spec.B)-slab_hamiltonian(p,float(y),float(k),spec.B))/Q
            R=U.conj().T@V; J=U.conj().T@G@V
            vr=max(vr,float(np.max(np.abs(J-(em[None,:]-en[:,None])*R/Q))))
            df=occ[iy,iz][:,None]-occ[iy,j][None,:]; c=df/(z+en[:,None]-em[None,:])
            K00+=(c*R*R.conj()).sum(); Kz0+=(c*J*R.conj()).sum(); K0z+=(c*R*J.conj()).sum(); Kzz+=(c*J*J.conj()).sum()
            cl+=(df*J*R.conj()).sum(); cr+=(df*R*J.conj()).sum()
    v=p.volume; contact=.5*(cl+cr)/(Q*v)
    K=np.array([[K00/v,K0z/v],[Kz0/v,Kzz/v+contact]],complex); qv=np.array([z,-Q])
    return WardResult(vr,float(np.max(np.abs(qv@K))),float(np.max(np.abs(K@qv))),contact,K)

def rpa_dress(K0:np.ndarray,V:complex):
    K=np.asarray(K0,complex); W=complex(V)/(1-complex(V)*K[0,0]); return K+np.outer(K[:,0],K[0,:])*W

# Rational-flux magnetic torus (all magnetic bands, no surface states).
@dataclass(frozen=True)
class TorusParams:
    Lx:int=8; Ly:int=8; Nkz:int=16; m:float=2.; tx:float=1.; ty:float=1.; tz:float=1.; vx:float=1.; vy:float=1.; theta_x:float=.271; theta_y:float=.413; theta_z:float=.173
    @property
    def nxy(self): return self.Lx*self.Ly
    @property
    def dim(self): return 2*self.nxy
    @property
    def volume(self): return self.nxy*self.Nkz
    def B(self,nphi): return 2*np.pi*nphi/self.nxy
    def kz_grid(self): return (2*np.pi*np.arange(self.Nkz)+self.theta_z)/self.Nkz-np.pi

def torus_xy(p:TorusParams,nphi:int):
    H=np.zeros((p.dim,p.dim),complex); B=p.B(nphi); Tx=-.5*p.tx*SZ-.5j*p.vx*SX; Ty=-.5*p.ty*SZ-.5j*p.vy*SY
    def sl(x,y): s=x*p.Ly+y; return slice(2*s,2*s+2)
    for x in range(p.Lx):
        for y in range(p.Ly):
            a=sl(x,y); H[a,a]+=p.m*SZ
            xp=(x+1)%p.Lx; b=sl(xp,y); ph=np.exp(1j*p.theta_x/p.Lx)*(np.exp(-1j*B*p.Lx*y) if x==p.Lx-1 else 1); h=ph*Tx; H[b,a]+=h; H[a,b]+=h.conj().T
            yp=(y+1)%p.Ly; b=sl(x,yp); ph=np.exp(1j*p.theta_y/p.Ly)*np.exp(1j*B*x); h=ph*Ty; H[b,a]+=h; H[a,b]+=h.conj().T
    return H

def torus_h(p:TorusParams,Hxy,kz): return Hxy+np.kron(np.eye(p.nxy),-p.tz*np.cos(kz)*SZ)
@dataclass
class TorusSpectrum:
    p:TorusParams; nphi:int; kz:np.ndarray; evals:np.ndarray; evecs:np.ndarray
    @property
    def B(self): return self.p.B(self.nphi)

def diagonalize_torus(p:TorusParams,nphi:int):
    Hxy=torus_xy(p,nphi); kz=p.kz_grid(); E=np.empty((p.Nkz,p.dim)); U=np.empty((p.Nkz,p.dim,p.dim),complex)
    for i,z in enumerate(kz): E[i],U[i]=sla.eigh(torus_h(p,Hxy,float(z)),check_finite=False)
    return TorusSpectrum(p,nphi,kz,E,U)

def torus_density(s:TorusSpectrum,mu,T): return float(np.sum(fermi(s.evals,mu,T))/s.p.volume)
def torus_mu(s:TorusSpectrum,target,T): return float(brentq(lambda m:torus_density(s,m,T)-target,s.evals.min()-1,s.evals.max()+1))

def torus_kubo(s:TorusSpectrum,mu,T,qshift,omega):
    p=s.p; sh=qshift%p.Nkz; occ=fermi(s.evals,mu,T); total=intra=inter=other=0j
    for i in range(p.Nkz):
        j=(i+sh)%p.Nkz; en=s.evals[i]; em=s.evals[j]; R=s.evecs[i].conj().T@s.evecs[j]
        t=(occ[i][:,None]-occ[j][None,:])*np.abs(R)**2/(complex(omega)+en[:,None]-em[None,:])
        total+=t.sum(); ep=en[:,None]>0; mp=em[None,:]>0; mi=ep&mp; mx=ep^mp
        intra+=t[mi].sum(); inter+=t[mx].sum(); other+=t[~(mi|mx)].sum()
    v=p.volume; return Components(total/v,intra/v,inter/v,other/v)

def torus_ward(s:TorusSpectrum,mu,T,qshift,omega):
    p=s.p; sh=qshift%p.Nkz; q=2*np.pi*sh/p.Nkz; Q=2*np.sin(q/2); z=complex(omega); occ=fermi(s.evals,mu,T); Hxy=torus_xy(p,s.nphi)
    K00=Kz0=K0z=Kzz=cl=cr=0j; vr=0.
    for i,k in enumerate(s.kz):
        j=(i+sh)%p.Nkz; kp=s.kz[j]; en=s.evals[i]; em=s.evals[j];U=s.evecs[i];V=s.evecs[j]
        G=(torus_h(p,Hxy,float(kp))-torus_h(p,Hxy,float(k)))/Q;R=U.conj().T@V;J=U.conj().T@G@V
        vr=max(vr,float(np.max(np.abs(J-(em[None,:]-en[:,None])*R/Q))))
        df=occ[i][:,None]-occ[j][None,:];c=df/(z+en[:,None]-em[None,:])
        K00+=(c*R*R.conj()).sum();Kz0+=(c*J*R.conj()).sum();K0z+=(c*R*J.conj()).sum();Kzz+=(c*J*J.conj()).sum();cl+=(df*J*R.conj()).sum();cr+=(df*R*J.conj()).sum()
    v=p.volume;contact=.5*(cl+cr)/(Q*v);K=np.array([[K00/v,K0z/v],[Kz0/v,Kzz/v+contact]],complex);qv=np.array([z,-Q])
    return WardResult(vr,float(np.max(np.abs(qv@K))),float(np.max(np.abs(K@qv))),contact,K)
