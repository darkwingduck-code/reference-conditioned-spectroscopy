"""Reference-assisted finite-window oscillator inference, with conditional error boxes.
All frequencies are dimensionless; omega is the laboratory frequency and
s=omega+i*eta is the retarded argument. No hidden frequency enters the estimator.
"""
from __future__ import annotations
import numpy as np
from oscillator_pilot import response,stiffness,modes,rational_fit,predict
from scipy.linalg import null_space

def transfer(omega):
    w=np.asarray(omega,complex)
    return 1.15*np.exp(.21j)*(w+.4+.5j)/(w+.6+.7j)

def standard(omega):
    return 1./((np.asarray(omega)+.04j)**2-1.7**2)

def blank(omega):
    return .004+.002j+(.006-.003j)*(np.asarray(omega)-.97)

def disk(rng,n,radius):
    return radius*np.sqrt(rng.uniform(size=n))*np.exp(2j*np.pi*rng.uniform(size=n))

def instrument_reference(omega,rng,calibration_radius):
    # These are separately measured known-standard and empty-path spectra.
    M=transfer(omega); a=blank(omega); std=standard(omega)
    refscale=float(np.sqrt(np.mean(abs(M*std)**2)))
    epsilon=calibration_radius*refscale
    bhat=a+disk(rng,len(omega),epsilon)
    ref=M*std+a+disk(rng,len(omega),epsilon)
    Mhat=(ref-bhat)/std
    epsM=2*epsilon/abs(std)
    return dict(Mhat=Mhat,bhat=bhat,epsM=epsM,epsblank=epsilon)

def calibrate(omega,measured,epsilon_measurement,reference):
    Mhat=reference["Mhat"]; bhat=reference["bhat"]
    chi=(measured-bhat)/Mhat
    lower=abs(Mhat)-reference["epsM"]
    eps=(epsilon_measurement+reference["epsblank"]+reference["epsM"]*abs(chi))/lower
    if np.any(lower<=0): raise ValueError("Detector inverse has no positive lower bound")
    # Strictly positive rounding allowance, declared separately from noise.
    eps+=5e-13*(1+abs(chi))
    return chi,eps

def measure(omega,truth,rng,noise_radius,reference):
    raw=transfer(omega)*truth+blank(omega)
    epsilon=noise_radius*float(np.sqrt(np.mean(abs(transfer(omega)*truth)**2)))
    y=raw+disk(rng,len(omega),epsilon)
    chi,eps=calibrate(omega,y,epsilon,reference)
    return chi,eps,epsilon

def infer_reference_zero(omega,eta,chi,eps_chi,chi0,eps_chi0):
    """Fit T=a*s^2+b with real (a,b); derive a worst-case parameter box.

    Certificate is conditional on the star model, unchanged bright/remote block,
    exactly specified common eta, and the supplied pointwise complex error bounds.
    No contour continuation or arbitrary-background certificate is implied.
    """
    chi=np.asarray(chi,complex);chi0=np.asarray(chi0,complex)
    eps_chi=np.asarray(eps_chi,float);eps_chi0=np.asarray(eps_chi0,float)
    omega=np.asarray(omega,float)
    reference_lower=abs(chi0)-eps_chi0
    A=1/chi0
    # Inversion bound expressed entirely in observed reference and its error radius.
    epsA=np.where(reference_lower>0,eps_chi0/(abs(chi0)*reference_lower),np.inf)
    D=A*chi-1
    epsD=abs(A)*eps_chi+abs(chi)*epsA+epsA*eps_chi
    margin=abs(D)-epsD
    valid=(reference_lower>0)&(margin>0)&np.isfinite(margin)
    result={"certified":False,"safe_samples":int(sum(valid)),"total_samples":len(omega),
       "unsafe_samples":int(sum(~valid)),"minimum_reference_lower":float(min(reference_lower)),
       "minimum_denominator_margin":float(min(margin)),"reason":""}
    # Data-only deterministic screening; excluded points are all counted.
    if sum(valid)<12:
        result["reason"]="fewer_than_12_safe_points";return result
    T=chi[valid]/D[valid]
    epsT=(eps_chi[valid]+epsA[valid]*abs(chi[valid])*(abs(chi[valid])+eps_chi[valid]))/(abs(D[valid])*margin[valid])
    s=omega[valid]+1j*eta
    Xc=np.column_stack([s*s,np.ones(len(s))])
    X=np.vstack([Xc.real,Xc.imag]);y=np.r_[T.real,T.imag]
    # Weight only by certified data error, never by hidden d/g.
    w=np.r_[1/epsT,1/epsT];w/=max(w)
    Xw=X*w[:,None];yw=y*w
    singular=np.linalg.svd(Xw,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float("inf")
    result.update(design_condition=condition,smallest_singular_value=float(singular[-1]),
                  largest_singular_value=float(singular[0]))
    if singular[-1]<=1e-12*singular[0] or condition>1e10:
        result["reason"]="rank_or_condition_safeguard";return result
    inverse=np.linalg.pinv(Xw)
    theta=inverse@yw
    influence=inverse*w[None,:]
    eps_real=np.r_[epsT,epsT]
    # Bound |Re e|,|Im e| separately by the complex radius. Conservative box.
    left_inverse_defect=influence@X-np.eye(2)
    defect_norm=float(np.linalg.norm(left_inverse_defect,ord=np.inf))
    result["left_inverse_defect_infinity_norm"]=defect_norm
    if defect_norm>=1e-8:
        result["reason"]="left_inverse_defect_safeguard";return result
    propagated_error=abs(influence)@eps_real
    theta_size=(max(abs(theta))+max(propagated_error))/(1-defect_norm)
    defect_allowance=abs(left_inverse_defect)@np.full(2,theta_size)
    arithmetic_padding=5e-12*(1+abs(theta))
    theta_error=propagated_error+defect_allowance+arithmetic_padding
    result.update(maximum_defect_allowance=float(max(defect_allowance)),
                  maximum_numerical_padding=float(max(arithmetic_padding)))
    # Exact-arithmetic inequalities plus numerically checked linear algebra.
    # The padding is not a formally outward-rounded interval-arithmetic proof.
    a,b=theta;da,db=theta_error
    residual=float(np.linalg.norm((X@theta-y)*w)/max(np.linalg.norm(y*w),1e-30))
    result.update(slope=float(a),intercept=float(b),slope_error=float(da),intercept_error=float(db),
       inferred_frequency=float(np.sqrt(-b/a)) if a>0 and b<0 else None,
       inferred_stiffness_coupling=float(1/np.sqrt(a)) if a>0 else None,
       transformed_relative_rms=residual,design_condition=float(np.linalg.cond(Xw)))
    if not(a-da>0 and b+db<0):
        result["reason"]="parameter_box_crosses_nonphysical_boundary";return result
    # Interval arithmetic over a positive-slope / negative-intercept rectangle.
    dsq=[-(b+ib*db)/(a+ia*da) for ia in (-1,1) for ib in (-1,1)]
    lo=float(np.sqrt(min(dsq)));hi=float(np.sqrt(max(dsq)))
    result.update(certified=True,frequency_lower=lo,frequency_upper=hi,
       maximum_frequency_error=float(max(result["inferred_frequency"]-lo,hi-result["inferred_frequency"])),
       reason="conditional_star_model_parameter_box")
    return result

def spectral_decomposition(q,B,h=.8):
    w,V=modes(q,B,h)
    W=abs(V[0])**2
    R=W[:2]/(2*w[:2])
    pp=float((R[0]*w[1]+R[1]*w[0])/sum(R))
    local_coordinate=float(np.sqrt((W[0]*w[1]**2+W[1]*w[0]**2)/sum(W[:2])))
    d=1.3*q
    return dict(pp_exact=pp,local_pair_coordinate_zero=local_coordinate,
       negative_frequency_correction=local_coordinate-pp,remote_pair_correction=d-local_coordinate,
       full_zero=d,low_pole=float(w[0]),high_local_pole=float(w[1]),remote_pole=float(w[2]),
       remote_weight=float(W[2]),local_weight_sum=float(sum(W[:2])))

def compressed_zeros(q,B,h=.8,source=None,**kw):
    v=np.array([1.,0.,0.]) if source is None else np.asarray(source,complex)
    Q=null_space(v.conj()[None,:])
    return np.sqrt(np.linalg.eigvalsh(Q.conj().T@stiffness(q,B,h,**kw)@Q))

