from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Embed TrueType outlines rather than Type-3 bitmap fonts in journal PDFs.
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'data_prl'/'full_bz_completion';F=ROOT/'fig_prl'/'full_bz_completion';F.mkdir(parents=True,exist_ok=True)
raw=pd.read_csv(D/'slab_full_band_raw.csv');fits=pd.read_csv(D/'slab_full_band_fits.csv');summ=json.loads((D/'slab_chain_rule_summary.json').read_text());conv=pd.read_csv(D/'slab_ky_convergence.csv')
# Fig S2
fig,ax=plt.subplots(2,2,figsize=(7.1,5.6))
dens=raw[raw.quantity=='density'].sort_values('B');x=dens.B.to_numpy()**2
ax[0,0].plot(x,dens.mu2q,'o',label='direct fixed-density solve');ax[0,0].axhline(summ['mu2_chain'],ls='--',label='chain rule')
ax[0,0].set(xlabel=r'$B^2$',ylabel=r'$(\mu(B)-\mu_0)/B^2$',title='(a) fixed-density ensemble');ax[0,0].legend(frameon=False,fontsize=7)
for ens,m in [('fixed_mu','o'),('fixed_n','s')]:
 s=raw[(raw.quantity=='response')&(raw.ensemble==ens)&(raw.component=='total')].sort_values('B');lab=r'fixed $\mu$' if ens=='fixed_mu' else r'fixed $n$';ax[0,1].plot(s.B**2,s.q2,m,label=lab);v=fits[(fits.ensemble==ens)&(fits.component=='total')].iloc[0].c2_re;ax[0,1].axhline(v,ls='--')
ax[0,1].set(xlabel=r'$B^2$',ylabel=r'$[K_{\rm even}(B)-K(0)]/B^2$',title='(b) full-band response');ax[0,1].legend(frameon=False,fontsize=7)
components=['conduction','interband','total'];xx=np.arange(3);w=.35
for j,ens in enumerate(['fixed_mu','fixed_n']):
 vals=[fits[(fits.ensemble==ens)&(fits.component==c)].iloc[0].c2_re for c in components];ax[1,0].bar(xx+(j-.5)*w,vals,w,label=(r'fixed $\mu$' if ens=='fixed_mu' else r'fixed $n$'))
ax[1,0].axhline(0,lw=.8);ax[1,0].set_xticks(xx,['intraband','interband','total']);ax[1,0].set(ylabel=r'$B^2$ coefficient',title='(c) filled-band/interband completion');ax[1,0].legend(frameon=False,fontsize=7)
ax[1,1].plot(conv.Nky,conv.K2_fixed_mu_real,'o-',label=r'fixed $\mu$');ax[1,1].plot(conv.Nky,conv.K2_fixed_n_real,'s-',label=r'fixed $n$');ax[1,1].set(xlabel=r'$N_{k_y}$',ylabel=r'extracted $B^2$ coefficient',title='(d) transverse-grid convergence');ax[1,1].legend(frameon=False,fontsize=7)
fig.tight_layout();fig.savefig(F/'figS2_full_band_ensemble.pdf');fig.savefig(F/'figS2_full_band_ensemble.png',dpi=220);plt.close(fig)

# Fig S3
ward=pd.read_csv(D/'slab_ward_rpa.csv');tward=pd.read_csv(D/'magnetic_torus_ward_rpa.csv');tc=pd.read_csv(D/'magnetic_torus_even_coefficients.csv');land=pd.read_csv(D/'exact_landau_spectrum.csv');lsum=json.loads((D/'exact_landau_projection_summary.json').read_text())
fig,ax=plt.subplots(2,2,figsize=(7.1,5.6))
for V,m in [(0.,'o'),(.7,'s'),(1.4,'^')]:
    ss=ward[ward.V==V]
    ax[0,0].semilogx(ss.B,ss.left,marker=m,label=rf'slab $V={V:g}$')
ax[0,0].semilogx(abs(tward.B),tward.left,'x',label='magnetic torus')
ax[0,0].set(xlabel=r'$|B|$',ylabel='left Ward residual',title='(a) bare and interacting transversality')
ax[0,0].legend(frameon=False,fontsize=6,ncol=2)
sub=tc[tc.component=='total']
for ens,m in [('fixed_mu','o'),('fixed_n','s')]:
    ss=sub[sub.ensemble==ens]
    ax[0,1].plot(ss.L,ss.c2_re,marker=m,linestyle='none',markersize=6,label=(r'fixed $\mu$' if ens=='fixed_mu' else r'fixed $n$'))
ax[0,1].axhline(0,lw=.8)
ax[0,1].set(xlabel='torus linear size',ylabel='finite-flux even coefficient',title='(b) no-surface magnetic torus')
ax[0,1].legend(frameon=False,fontsize=7)
ax[0,1].text(.04,.96,r'left/right Ward residual $<4.3\times10^{-17}$',transform=ax[0,1].transAxes,va='top',fontsize=7)
sel=land.iloc[:12]
ax[1,0].plot(sel['mode'],sel['eigenvalue'],'o-',label='Landau eigenvalue')
axr=ax[1,0].twinx()
axr.plot(sel['mode'],sel['charge_overlap'],'s--',label='charge overlap')
axr.plot(sel['mode'],sel['axial_overlap'],'^:',label='axial overlap')
ax[1,0].axhline(0,lw=.8)
ax[1,0].set(xlabel='patch-mode index',ylabel='dimensionless eigenvalue',title=r'(c) $Z_iZ_j\Gamma^\omega$ patch operator')
axr.set_ylabel('mode overlap')
h1,l1=ax[1,0].get_legend_handles_labels();h2,l2=axr.get_legend_handles_labels();ax[1,0].legend(h1+h2,l1+l2,frameon=False,fontsize=6,loc='center right')
vals=[lsum['U1_projection_residual'],lsum['U1_eigenvalue_residual'],lsum['nonabelian_covariance_residual'],lsum['nonabelian_energy_residual']]
names=[r'$U(1)$ tensor',r'$U(1)$ spectrum',r'$U(2)$ tensor',r'$U(2)$ energy']
ax[1,1].bar(np.arange(4),vals);ax[1,1].set_yscale('log');ax[1,1].set_xticks(np.arange(4),names,rotation=25,ha='right')
ax[1,1].set(ylabel='gauge-covariance residual',title='(d) band-gauge invariance')
ax[1,1].text(.04,.94,rf'two-mode leakage = {100*lsum["relative_two_mode_leakage"]:.1f}%',transform=ax[1,1].transAxes,va='top',fontsize=7)
fig.tight_layout();fig.savefig(F/'figS3_ward_landau_completion.pdf');fig.savefig(F/'figS3_ward_landau_completion.png',dpi=220);plt.close(fig)
