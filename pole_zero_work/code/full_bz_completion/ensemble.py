"""Exact weak-field conversion between fixed chemical potential and fixed density."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class MuExpansion:
    mu1: float
    mu2: float

def fixed_density_mu(*, n0p: float, n0pp: float, n1: float, n1p: float, n2: float) -> MuExpansion:
    """For n(mu,B)=n0(mu)+B n1(mu)+B^2 n2(mu)+..., return mu1,mu2."""
    if not np.isfinite(n0p) or abs(n0p) == 0:
        raise ValueError("n0p must be finite and nonzero")
    mu1 = -n1/n0p
    mu2 = -(n2 + mu1*n1p + 0.5*mu1**2*n0pp)/n0p
    return MuExpansion(float(mu1), float(mu2))

def fixed_density_response(*, K0: Any, K0p: Any, K0pp: Any,
                           K1: Any, K1p: Any, K2: Any,
                           mu1: float, mu2: float):
    """For K(mu,B)=K0+B K1+B^2 K2+..., return fixed-density K0,K1,K2."""
    return (
        np.asarray(K0),
        np.asarray(K1) + mu1*np.asarray(K0p),
        np.asarray(K2) + mu1*np.asarray(K1p) + mu2*np.asarray(K0p)
        + 0.5*mu1**2*np.asarray(K0pp),
    )

def hall_compensated(*, n0p: float, n2: float, K0p: Any, K2: Any):
    """Special case n1=mu1=0."""
    if not np.isfinite(n0p) or abs(n0p) == 0:
        raise ValueError("n0p must be finite and nonzero")
    mu2 = -float(n2)/float(n0p)
    return mu2, np.asarray(K2)+mu2*np.asarray(K0p)
