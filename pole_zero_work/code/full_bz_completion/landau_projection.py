"""Microscopic Landau projection from Sigma and the dynamic forward vertex Gamma^omega."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class FermiSurface:
    states:np.ndarray; area:np.ndarray; velocity:np.ndarray; residue:np.ndarray; node:np.ndarray
    def __post_init__(self):
        u=np.asarray(self.states,complex); n=u.shape[0]
        if u.ndim!=2 or any(np.asarray(x).shape!=(n,) for x in (self.area,self.velocity,self.residue,self.node)): raise ValueError("shape mismatch")
        if np.max(np.abs(np.sum(abs(u)**2,axis=1)-1))>1e-10: raise ValueError("states not normalized")
        if np.any(np.asarray(self.area)<=0) or np.any(np.asarray(self.velocity)<=0) or np.any(np.asarray(self.residue)<=0): raise ValueError("nonpositive input")
    @property
    def n(self): return self.states.shape[0]
    @property
    def norb(self): return self.states.shape[1]
    @property
    def weights(self): return np.asarray(self.area)/((2*np.pi)**3*np.asarray(self.velocity))

def residue_from_sigma(dsigma_domega):
    z=1/(1-np.asarray(dsigma_domega,float));
    if np.any(z<=0): raise ValueError("unstable residue")
    return z

def velocity_from_sigma(v0,grad_sigma,Z): return np.asarray(Z)[:,None]*(np.asarray(v0)+np.asarray(grad_sigma))

def project_gamma(fs:FermiSurface,Gamma,hermitize=True):
    G=np.asarray(Gamma,complex); exp=(fs.n,fs.n,fs.norb,fs.norb,fs.norb,fs.norb)
    if G.shape!=exp: raise ValueError(f"Gamma must have shape {exp}")
    u=np.asarray(fs.states,complex); z=np.asarray(fs.residue,float); f=np.empty((fs.n,fs.n),complex)
    for i in range(fs.n):
        for j in range(fs.n): f[i,j]=z[i]*z[j]*np.einsum('a,b,abcd,c,d->',u[i].conj(),u[j].conj(),G[i,j],u[i],u[j],optimize=True)
    return .5*(f+f.conj().T) if hermitize else f

def landau_operator(fs:FermiSurface,f):
    sw=np.sqrt(fs.weights); h=.5*(np.asarray(f)+np.asarray(f).conj().T); return sw[:,None]*h*sw[None,:]

def mode_diagnostics(fs:FermiSurface,f,signs:dict[int,float]):
    L=landau_operator(fs,f); val,vec=np.linalg.eigh(L); o=np.argsort(val)[::-1]; val=val[o];vec=vec[:,o]
    c=np.sqrt(fs.weights); c=c/np.linalg.norm(c); a=np.sqrt(fs.weights)*np.array([signs[int(x)] for x in fs.node]); a-=c*(c.conj()@a);a/=np.linalg.norm(a)
    B=np.column_stack([c,a]); leak=np.linalg.norm((np.eye(fs.n)-B@B.conj().T)@L@B,'fro')/np.linalg.norm(L@B,'fro')
    return dict(eigenvalues=val,eigenvectors=vec,charge_overlap=abs(c.conj()@vec)**2,axial_overlap=abs(a.conj()@vec)**2,two_mode=B.conj().T@L@B,relative_leakage=float(leak))

def apply_phases(fs:FermiSurface,phi):
    return FermiSurface(np.asarray(fs.states)*np.exp(1j*np.asarray(phi))[:,None],fs.area,fs.velocity,fs.residue,fs.node)

def _sqrt_pos(a):
    x=.5*(np.asarray(a,complex)+np.asarray(a,complex).conj().T); v,u=np.linalg.eigh(x)
    if v.min()<-1e-12: raise ValueError("Z not positive")
    return (u*np.sqrt(np.clip(v,0,None))[None,:])@u.conj().T

def project_subspaces(frames,Zmat,Gamma):
    U=np.asarray(frames,complex);Z=np.asarray(Zmat,complex);G=np.asarray(Gamma,complex);P,A,R=U.shape
    if Z.shape!=(P,R,R) or G.shape!=(P,P,A,A,A,A): raise ValueError("shape mismatch")
    V=np.empty_like(U)
    for i in range(P): V[i]=U[i]@_sqrt_pos(Z[i])
    f=np.empty((P,P,R,R,R,R),complex)
    for i in range(P):
        for j in range(P): f[i,j]=np.einsum('am,bp,abcd,cn,dq->mnpq',V[i].conj(),V[j].conj(),G[i,j],V[i],V[j],optimize=True)
    return f

def rotate_subspaces(U,Z,W):
    return np.einsum('iam,imn->ian',U,W),np.einsum('ima,imn,inb->iab',W.conj(),Z,W)
def rotate_projected(f,W):
    P=f.shape[0];out=np.empty_like(f)
    for i in range(P):
        for j in range(P): out[i,j]=np.einsum('ma,nb,pc,qd,mnpq->abcd',W[i].conj(),W[i],W[j].conj(),W[j],f[i,j],optimize=True)
    return out
def rotate_density(N,W): return np.einsum('ima,imn,inb->iab',W.conj(),N,W)
def subspace_energy(f,N,w=None):
    if w is None:w=np.ones(f.shape[0])
    return .5*np.einsum('i,j,ijmnpq,inm,jqp->',w,w,f,N,N,optimize=True)
