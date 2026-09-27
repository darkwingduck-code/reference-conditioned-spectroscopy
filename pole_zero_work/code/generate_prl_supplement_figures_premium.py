#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data_prl'
FIG = ROOT / 'fig_prl'
FULLBZ_DATA = DATA / 'full_bz_completion'
FULLBZ_FIG = FIG / 'full_bz_completion'
FIG.mkdir(parents=True, exist_ok=True)
FULLBZ_FIG.mkdir(parents=True, exist_ok=True)

INK = '#18232E'
NAVY = '#244A64'
TEAL = '#2A7F78'
VERMILION = '#C65A46'
OCHRE = '#B7862D'
PLUM = '#70566E'
SLATE = '#687682'
MID_GREY = '#9AA4AD'
GRID = '#DDE2E6'
PALE = '#F3F5F7'
PALE_BLUE = '#EEF3F6'
PALE_ROSE = '#F8F0EF'
WHITE = '#FFFFFF'

mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Noto Sans'],
    'mathtext.fontset': 'dejavusans',
    'font.size': 7.8,
    'axes.labelsize': 8.0,
    'axes.titlesize': 8.3,
    'axes.titleweight': 'semibold',
    'axes.linewidth': 0.72,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'xtick.labelsize': 7.1,
    'ytick.labelsize': 7.1,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3.2,
    'ytick.major.size': 3.2,
    'xtick.minor.size': 1.8,
    'ytick.minor.size': 1.8,
    'xtick.major.width': 0.70,
    'ytick.major.width': 0.70,
    'xtick.minor.width': 0.55,
    'ytick.minor.width': 0.55,
    'legend.fontsize': 6.6,
    'legend.frameon': False,
    'figure.facecolor': WHITE,
    'axes.facecolor': WHITE,
    'savefig.facecolor': WHITE,
    'axes.unicode_minus': False,
})


def polish(ax, ygrid=False, xgrid=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(top=False, right=False)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis='y', color=GRID, lw=0.5)
    if xgrid:
        ax.grid(axis='x', color=GRID, lw=0.5)


def label_panel(ax, label, title):
    ax.set_title(rf'$\bf{{({label})}}$  {title}', loc='left', pad=6, color=INK)


def savefig(fig, outpath: Path, title: str):
    meta = {'Title': title, 'Creator': 'Matplotlib'}
    fig.savefig(outpath.with_suffix('.pdf'), bbox_inches='tight', metadata=meta)
    fig.savefig(outpath.with_suffix('.png'), bbox_inches='tight', dpi=320)
    plt.close(fig)


def figS1():
    disp = pd.read_csv(DATA / 'figS1_alternative_dispersion.csv')
    mp = pd.read_csv(DATA / 'figS1_interaction_crossing_map.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))

    # Panel (a): the physically relevant two-dimensional map is (F0a,F1a).
    # F0x=(F0c-F0a)/2 is derived from F0a and must not be used as a second
    # independent map axis.
    ax = axes[0]
    polish(ax)
    pivot = mp.pivot_table(index='F1a', columns='F0a', values='qbar_crossing', aggfunc='mean')
    xvals = pivot.columns.to_numpy(float)
    yvals = pivot.index.to_numpy(float)
    im = ax.imshow(
        pivot.values,
        origin='lower',
        aspect='auto',
        extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
        cmap='cividis_r',
        interpolation='nearest',
    )
    # F0x=0 corresponds to F0a=F0c=1 for this scan.
    ax.axvline(1.0, color=WHITE, lw=1.0, ls=(0, (3.0, 2.2)))
    ax.text(1.0, yvals.max(), r'$F_{0x}=0$', color=WHITE, fontsize=6.0,
            ha='center', va='top', rotation=90,
            bbox={'boxstyle':'round,pad=0.12','fc':INK,'ec':'none','alpha':0.55})
    alt_idx = ((mp.F0a - 0.8).abs() + (mp.F1a - 1.0).abs()).idxmin()
    alt = mp.loc[[alt_idx]]
    ax.scatter(alt.F0a, alt.F1a, marker='*', s=72, color=VERMILION,
               edgecolors=WHITE, linewidths=0.7, zorder=5, label='alternative point')
    ax.contour(xvals, yvals, pivot.values, levels=[0.2], colors=[WHITE], linewidths=0.9)
    ax.set(xlabel=r'$F_{0a}$', ylabel=r'$F_{1a}$')
    label_panel(ax, 'a', 'Finite crossing across interaction space')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r'$\bar q_{\rm cross}$')
    ax.legend(loc='lower right', fontsize=5.9)

    ax = axes[1]
    polish(ax)
    q = np.sort(disp.qbar.unique())
    for branch, color, ls, lw, lab in [
        ('charge', NAVY, (0, (5, 2.4)), 1.5, r'charge, $b=0$'),
        ('axial', OCHRE, (0, (1.2, 1.8)), 1.5, r'alternative axial, $b=0$'),
        ('lower', PLUM, '-', 1.9, r'hybrid poles, $b=0.01$'),
        ('upper', PLUM, '-', 1.9, None),
    ]:
        sub = disp[disp.branch == branch].sort_values('qbar')
        ax.plot(sub.qbar, sub.omega_bar, color=color, ls=ls, lw=lw, label=lab)
    charge = disp[disp.branch == 'charge'].sort_values('qbar')
    axial = disp[disp.branch == 'axial'].sort_values('qbar')
    q0 = charge.iloc[np.argmin(np.abs(charge.omega_bar.to_numpy() - axial.omega_bar.to_numpy()))].qbar
    ax.axvline(q0, color=MID_GREY, lw=0.8, ls=(0, (3, 2, 1, 2)))
    ax.fill_between(q, 0, q, color=PALE, zorder=0)
    ax.plot(q, q, color=MID_GREY, lw=0.7, ls=(0, (3, 2)))
    ax.set(xlabel=r'$\bar q$', ylabel=r'$\bar\omega$')
    label_panel(ax, 'b', 'Avoided crossing at the alternative point')
    ax.legend(loc='upper left')

    fig.subplots_adjust(wspace=0.36)
    savefig(fig, FIG / 'figS1_interaction_robustness', 'Interaction robustness')

def figS4():
    noise = pd.read_csv(DATA / 'figS4_complex_tomography_noise.csv')
    third = pd.read_csv(DATA / 'figS4_three_mode_diagnostic.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))

    ax = axes[0]
    polish(ax, ygrid=True)
    grouped = noise.groupby('snr')
    snr = np.array(sorted(grouped.groups))
    med = grouped['zero_error'].median().reindex(snr).to_numpy()
    q10 = grouped['zero_error'].quantile(0.1).reindex(snr).to_numpy()
    q90 = grouped['zero_error'].quantile(0.9).reindex(snr).to_numpy()
    rms = grouped['relative_fit_rms'].median().reindex(snr).to_numpy()
    ax.fill_between(snr, q10, q90, color=PALE_BLUE, zorder=1)
    ax.plot(snr, med, color=NAVY, lw=1.7, marker='o', ms=3.8, mfc=WHITE, mec=NAVY, label='median zero error')
    ax.plot(snr, rms, color=TEAL, lw=1.4, marker='s', ms=3.6, mfc=WHITE, mec=TEAL, label='median fit RMS')
    ax.set_xscale('log')
    ax.set_xlabel('complex-response SNR')
    ax.set_ylabel('relative reconstruction error')
    label_panel(ax,'a','Noise robustness of complex tomography')
    ax.legend(loc='upper right')

    ax = axes[1]
    polish(ax)
    sc = ax.scatter(third.third_pole_distance, third.third_pole_strength,
                    c=np.log10(np.maximum(third.zero_error,1e-5)), cmap='magma_r',
                    s=28 + 60*third.relative_two_pole_fit_rms/third.relative_two_pole_fit_rms.max(),
                    edgecolors=WHITE, linewidths=0.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('distance to third pole')
    ax.set_ylabel('third-pole strength')
    label_panel(ax,'b','Breakdown against hidden third-mode contamination')
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r'$\log_{10}$ zero error')
    fig.subplots_adjust(wspace=0.33)
    savefig(fig, FIG / 'figS4_complex_tomography', 'Complex tomography robustness')


def figS5():
    ms = pd.read_csv(DATA / 'figS5_genealogy_model_selection.csv')
    rep = pd.read_csv(DATA / 'figS5_genealogy_representative_fits.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))

    ax = axes[0]
    polish(ax, ygrid=True)
    for truth, color, marker in [('finite', TEAL, 'o'), ('vanishing', VERMILION, 's')]:
        sub = ms[(ms.truth == truth) & (np.isclose(ms.relative_q_noise, 0.01))].sort_values('field_span')
        ax.plot(sub.field_span, sub.accuracy, color=color, lw=1.7, marker=marker, ms=4.0,
                mfc=WHITE, mec=color, label=f'{truth} truth')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('field span factor')
    ax.set_ylabel('classification accuracy')
    label_panel(ax,'a','Finite-window genealogy inference')
    ax.legend(loc='lower right')

    ax = axes[1]
    polish(ax)
    for truth, color in [('finite', TEAL), ('vanishing', VERMILION)]:
        sub = rep[rep.truth == truth].sort_values('abs_b')
        ax.errorbar(sub.abs_b, sub.q_observed, yerr=sub.sigma, fmt='o', ms=3.4, color=color,
                    mfc=WHITE, mec=color, lw=1.0, capsize=2, label=f'observed ({truth})')
        ax.plot(sub.abs_b, sub.finite_fit, color=NAVY if truth=='finite' else PLUM, lw=1.2,
                ls='-', alpha=0.9)
        ax.plot(sub.abs_b, sub.exact_vanishing_fit, color=OCHRE, lw=1.0, ls=(0,(2,2)), alpha=0.85)
    ax.set_xscale('log')
    ax.set_xlabel(r'$|b_\parallel|$')
    ax.set_ylabel(r'$q_{\rm cross}$')
    label_panel(ax,'b','Representative finite-window fits')
    handles = [
        mpl.lines.Line2D([],[], color=TEAL, marker='o', mfc=WHITE, mec=TEAL, lw=0, label='observed finite'),
        mpl.lines.Line2D([],[], color=VERMILION, marker='o', mfc=WHITE, mec=VERMILION, lw=0, label='observed vanishing'),
        mpl.lines.Line2D([],[], color=NAVY, lw=1.2, label='finite-intercept fit'),
        mpl.lines.Line2D([],[], color=OCHRE, lw=1.0, ls=(0,(2,2)), label='exact vanishing fit'),
    ]
    ax.legend(handles=handles, loc='best', fontsize=6.0)
    fig.subplots_adjust(wspace=0.34)
    savefig(fig, FIG / 'figS5_genealogy_inference', 'Genealogy inference')


def figS6():
    df = pd.read_csv(DATA / 'figS6_symmetry_selection_rules.csv').sort_values('b')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    ax = axes[0]
    polish(ax, ygrid=True)
    ax.plot(df.b, df.qmin_node_exchange, color=TEAL, lw=1.6, label='node-exchange symmetric')
    ax.plot(df.b, df.qmin_symmetry_broken, color=VERMILION, lw=1.6, label='symmetry-broken')
    ax.set_xscale('log')
    ax.set_xlabel(r'$|b|$')
    ax.set_ylabel(r'$q_{\min}$')
    label_panel(ax,'a',r'Intercept drift with and without symmetry')
    ax.legend(loc='best')

    ax = axes[1]
    polish(ax, ygrid=True)
    ax.plot(df.b, df.gap_linear_mixing, color=NAVY, lw=1.6, label=r'linear mixing, $\nu=1$')
    ax.plot(df.b, df.gap_quadratic_mixing, color=OCHRE, lw=1.6, ls=(0,(2,2)), label=r'quadratic mixing, $\nu=2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$|b|$')
    ax.set_ylabel(r'$\Delta\omega_{\min}$')
    label_panel(ax,'b',r'Selection-rule control of gap scaling')
    ax.legend(loc='best')
    fig.subplots_adjust(wspace=0.34)
    savefig(fig, FIG / 'figS6_symmetry_selection_rules', 'Symmetry selection rules')


def figS7():
    pz = pd.read_csv(DATA / 'figS7_multimode_poles_zeros.csv')
    loss = pd.read_csv(DATA / 'figS7_loss_only_ambiguity.csv')
    cof = pd.read_csv(DATA / 'figS7_cofactor_jacobi_residuals.csv')
    bg = pd.read_csv(DATA / 'figS7_background_dependent_total_zeros.csv')
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.55))

    ax = axes[0, 0]
    polish(ax)
    poles = pz[pz.kind == 'hybrid_pole']
    zeros = pz[pz.kind == 'response_zero']
    dark = pz[pz.kind == 'dark_block_eigenvalue']
    ax.scatter(poles.real, poles.imag, color=NAVY, marker='o', s=30,
               edgecolors=WHITE, linewidths=0.45, label='hybrid poles')
    # Open diamonds and smaller filled points make the coincidence visible.
    ax.scatter(dark.real, dark.imag, facecolors='none', edgecolors=TEAL,
               marker='D', s=45, linewidths=1.15, label='dark-block roots')
    ax.scatter(zeros.real, zeros.imag, color=VERMILION, marker='x', s=28,
               linewidths=1.1, label='auto-response zeros')
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel(r'$\Re\,\omega$')
    ax.set_ylabel(r'$\Im\,\omega$')
    label_panel(ax, 'a', 'Pole-zero structure in a multimode response')
    ax.legend(loc='best', fontsize=5.8)

    ax = axes[0, 1]
    polish(ax)
    ax.plot(loss.omega, loss.minus_im_chi_background_1, color=NAVY, lw=1.5,
            label=r'$-\Im\chi$, background 1')
    ax.plot(loss.omega, loss.minus_im_chi_background_2, color=VERMILION,
            lw=1.2, ls=(0, (2, 2)), label=r'$-\Im\chi$, background 2')
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel('loss response')
    label_panel(ax, 'b', 'Identical finite-window loss')
    ax.legend(loc='best', fontsize=5.8)

    ax = axes[1, 0]
    polish(ax, ygrid=True)
    ax.plot(cof.matrix_size,
            np.maximum(cof.max_scalar_cofactor_relative_residual, 1e-18),
            color=TEAL, lw=1.5, marker='o', mfc=WHITE, mec=TEAL,
            label='scalar cofactor')
    jp = cof.max_two_probe_jacobi_relative_residual.fillna(np.nan)
    ax.plot(cof.matrix_size, np.maximum(jp, 1e-18), color=PLUM, lw=1.5,
            marker='s', mfc=WHITE, mec=PLUM, label='two-probe Jacobi')
    ax.set_yscale('log')
    ax.set_xlabel('matrix size')
    ax.set_ylabel('relative residual')
    label_panel(ax, 'c', 'Cofactor and Jacobi identities')
    ax.legend(loc='best')

    ax = axes[1, 1]
    polish(ax)
    total = bg[bg.kind == 'total_response_zero']
    ax.scatter(total.real, total.imag, color=VERMILION, marker='o', s=30,
               label='total-response zeros')
    sing = bg[bg.kind == 'singular_zero']
    ax.scatter(sing.real, sing.imag, color=TEAL, marker='D', s=40,
               label='principal-part zero')
    # Label unique points only; duplicate background labels otherwise obscure the panel.
    for idx, row in bg.reset_index(drop=True).iterrows():
        label = str(row.background).replace('_', ' ')
        if row.kind == 'singular_zero':
            label = 'subtracted'
        ax.annotate(label, (row.real, row.imag), xytext=(5, 4),
                    textcoords='offset points', fontsize=5.5, color=SLATE)
    ax.set_xlabel(r'$\Re\,\omega$')
    ax.set_ylabel(r'$\Im\,\omega$')
    label_panel(ax, 'd', 'Background-dependent total-response zeros')
    ax.legend(loc='upper right', fontsize=5.5)

    fig.subplots_adjust(wspace=0.34, hspace=0.42)
    savefig(fig, FIG / 'figS7_multimode_and_identifiability',
            'Multimode tomography and identifiability')

def figS8():
    df = pd.read_csv(DATA / 'figS8_tomography_observability_map.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.95))
    ratio = np.sort(df.gap_to_sum_halfwidth_ratio.unique())
    snr = np.sort(df.complex_response_snr.unique())
    Z1 = df.pivot(index='gap_to_sum_halfwidth_ratio', columns='complex_response_snr', values='probability_zero_error_below_0p2gamma').reindex(index=ratio, columns=snr)
    Z2 = df.pivot(index='gap_to_sum_halfwidth_ratio', columns='complex_response_snr', values='median_zero_error_over_gamma').reindex(index=ratio, columns=snr)

    ax = axes[0]
    polish(ax)
    im = ax.imshow(Z1.values, origin='lower', aspect='auto', cmap='viridis',
                   extent=[snr.min(), snr.max(), ratio.min(), ratio.max()])
    ax.set_xscale('log')
    ax.set_xlabel('complex-response SNR')
    ax.set_ylabel(r'$\Delta\omega /(\gamma_++\gamma_-)$')
    label_panel(ax,'a',r'Probability of sub-$0.2\gamma$ zero reconstruction')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('success probability')

    ax = axes[1]
    polish(ax)
    im2 = ax.imshow(Z2.values, origin='lower', aspect='auto', cmap='magma_r',
                    extent=[snr.min(), snr.max(), ratio.min(), ratio.max()])
    ax.set_xscale('log')
    ax.set_xlabel('complex-response SNR')
    ax.set_ylabel(r'$\Delta\omega /(\gamma_++\gamma_-)$')
    label_panel(ax,'b',r'Median zero error in units of linewidth')
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
    cbar2.set_label(r'median zero error / $\gamma$')
    fig.subplots_adjust(wspace=0.34)
    savefig(fig, FIG / 'figS8_tomography_observability_map', 'Tomography observability map')


def figS9():
    df = pd.read_csv(DATA / 'figS9_competitor_exact_trajectory.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85))
    ax = axes[0]
    polish(ax)
    ax.plot(df.x_vOmega_qTF_over_sqrt_eps_omegaP, df.normalized_anomaly_crossing_q, color=VERMILION, lw=1.8, label='exact competitor crossover')
    ax.plot(df.x_vOmega_qTF_over_sqrt_eps_omegaP, df.weak_field_1_over_x, color=OCHRE, lw=1.1, ls=(0,(2,2)), label=r'weak-field $1/x$ asymptote')
    ax.plot(df.x_vOmega_qTF_over_sqrt_eps_omegaP, df.representative_finite_intercept_q_over_q0, color=TEAL, lw=1.5, label='finite-intercept reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$x=v\Omega q_{\rm TF}/(\sqrt{\varepsilon}\,\omega_P)$')
    ax.set_ylabel(r'$q/q_0$')
    label_panel(ax,'a','Exact crossover of the competitor mechanism')
    ax.legend(loc='best')

    ax = axes[1]
    polish(ax, ygrid=True)
    ax.semilogx(df.x_vOmega_qTF_over_sqrt_eps_omegaP, df.anomaly_effective_zeta, color=VERMILION, lw=1.8, label='competitor local exponent')
    ax.semilogx(df.x_vOmega_qTF_over_sqrt_eps_omegaP, df.representative_finite_intercept_zeta, color=TEAL, lw=1.5, label='finite-intercept reference')
    ax.axhline(1.0, color=MID_GREY, lw=0.8, ls=(0,(3,2)))
    ax.axhline(0.0, color=MID_GREY, lw=0.8, ls=(0,(3,2)))
    ax.set_xlabel(r'$x=v\Omega q_{\rm TF}/(\sqrt{\varepsilon}\,\omega_P)$')
    ax.set_ylabel(r'$\zeta=-d\ln q_{\rm cross}/d\ln|B|$')
    label_panel(ax,'b','Local genealogy exponent')
    ax.legend(loc='best')
    fig.subplots_adjust(wspace=0.34)
    savefig(fig, FIG / 'figS9_competitor_exact_trajectory', 'Exact competitor crossover')


def figS2S3_fullbz():
    raw = pd.read_csv(FULLBZ_DATA/'slab_full_band_raw.csv')
    fits = pd.read_csv(FULLBZ_DATA/'slab_full_band_fits.csv')
    summ = json.loads((FULLBZ_DATA/'slab_chain_rule_summary.json').read_text())
    conv = pd.read_csv(FULLBZ_DATA/'slab_ky_convergence.csv')

    fig, ax = plt.subplots(2, 2, figsize=(7.15, 5.55))
    # S2a
    dens = raw[raw.quantity=='density'].sort_values('B')
    x = dens.B.to_numpy()**2
    a = ax[0,0]; polish(a, ygrid=True)
    a.plot(x, dens.mu2q, color=NAVY, marker='o', ms=3.3, mfc=WHITE, mec=NAVY, lw=1.4, label='direct fixed-density solve')
    a.axhline(summ['mu2_chain'], color=OCHRE, lw=1.2, ls=(0,(2,2)), label='chain rule')
    a.set(xlabel=r'$B^2$', ylabel=r'$(\mu(B)-\mu_0)/B^2$'); label_panel(a,'a','Fixed-density ensemble correction'); a.legend(loc='best', fontsize=5.9)
    # S2b
    a = ax[0,1]; polish(a, ygrid=True)
    for ens, color, marker, lab in [('fixed_mu', TEAL, 'o', r'fixed $\mu$'), ('fixed_n', VERMILION, 's', r'fixed $n$')]:
        s = raw[(raw.quantity=='response') & (raw.ensemble==ens) & (raw.component=='total')].sort_values('B')
        a.plot(s.B**2, s.q2, color=color, marker=marker, ms=3.3, mfc=WHITE, mec=color, lw=1.4, label=lab)
        v = fits[(fits.ensemble==ens) & (fits.component=='total')].iloc[0].c2_re
        a.axhline(v, color=color, lw=0.9, ls=(0,(2,2)), alpha=0.9)
    a.set(xlabel=r'$B^2$', ylabel=r'$[K_{\rm even}(B)-K(0)]/B^2$'); label_panel(a,'b','Full-band even response'); a.legend(loc='best', fontsize=5.9)
    # S2c
    a = ax[1,0]; polish(a, ygrid=True)
    components = ['conduction','interband','total']
    xx = np.arange(3); w=.34
    for j,(ens, color, lab) in enumerate([('fixed_mu', TEAL, r'fixed $\mu$'), ('fixed_n', VERMILION, r'fixed $n$')]):
        vals = [fits[(fits.ensemble==ens) & (fits.component==c)].iloc[0].c2_re for c in components]
        a.bar(xx+(j-.5)*w, vals, w, color=color if ens=='fixed_n' else PALE_BLUE, edgecolor=color, linewidth=0.8, label=lab)
    a.axhline(0, color=MID_GREY, lw=0.8)
    a.set_xticks(xx, ['intraband','interband','total'])
    a.set_ylabel(r'$B^2$ coefficient'); label_panel(a,'c','Filled-band and interband completion'); a.legend(loc='best', fontsize=5.9)
    # S2d
    a = ax[1,1]; polish(a, ygrid=True)
    a.plot(conv.Nky, conv.K2_fixed_mu_real, color=TEAL, marker='o', ms=3.2, mfc=WHITE, mec=TEAL, lw=1.4, label=r'fixed $\mu$')
    a.plot(conv.Nky, conv.K2_fixed_n_real, color=VERMILION, marker='s', ms=3.2, mfc=WHITE, mec=VERMILION, lw=1.4, label=r'fixed $n$')
    a.set(xlabel=r'$N_{k_y}$', ylabel=r'extracted $B^2$ coefficient'); label_panel(a,'d','Transverse-grid convergence'); a.legend(loc='best', fontsize=5.9)
    fig.subplots_adjust(wspace=0.34, hspace=0.42)
    savefig(fig, FULLBZ_FIG / 'figS2_full_band_ensemble', 'Full-band ensemble completion')

    ward = pd.read_csv(FULLBZ_DATA/'slab_ward_rpa.csv')
    tward = pd.read_csv(FULLBZ_DATA/'magnetic_torus_ward_rpa.csv')
    tc = pd.read_csv(FULLBZ_DATA/'magnetic_torus_even_coefficients.csv')
    land = pd.read_csv(FULLBZ_DATA/'exact_landau_spectrum.csv')
    lsum = json.loads((FULLBZ_DATA/'exact_landau_projection_summary.json').read_text())
    fig, ax = plt.subplots(2, 2, figsize=(7.15, 5.55))
    # S3a
    a = ax[0,0]; polish(a, ygrid=True)
    for V, color, marker in [(0., NAVY, 'o'), (.7, TEAL, 's'), (1.4, VERMILION, '^')]:
        ss = ward[ward.V==V]
        a.semilogx(ss.B, ss.left, color=color, marker=marker, ms=3.2, mfc=WHITE, mec=color, lw=1.2, label=rf'slab $V={V:g}$')
    a.semilogx(abs(tward.B), tward.left, color=PLUM, marker='D', ls='none', ms=3.0, label='magnetic torus')
    a.set(xlabel=r'$|B|$', ylabel='left Ward residual'); label_panel(a,'a','Bare and interacting transversality'); a.legend(loc='best', fontsize=5.7, ncol=2)
    # S3b
    a = ax[0,1]; polish(a, ygrid=True)
    sub = tc[tc.component=='total']
    for ens, color, marker, lab in [('fixed_mu', TEAL, 'o', r'fixed $\mu$'), ('fixed_n', VERMILION, 's', r'fixed $n$')]:
        ss = sub[sub.ensemble==ens]
        a.plot(ss.L, ss.c2_re, color=color, marker=marker, ms=3.2, mfc=WHITE, mec=color, lw=1.2, label=lab)
    a.axhline(0, color=MID_GREY, lw=0.8)
    a.text(0.03, 0.95, r'Ward residual $<4.3\times10^{-17}$', transform=a.transAxes, va='top', fontsize=6.0, color=SLATE)
    a.set(xlabel='torus linear size', ylabel='finite-flux even coefficient'); label_panel(a,'b','No-surface magnetic torus'); a.legend(loc='best', fontsize=5.9)
    # S3c
    a = ax[1,0]; polish(a)
    sel = land.iloc[:12]
    a.plot(sel['mode'], sel['eigenvalue'], color=NAVY, marker='o', ms=3.2, mfc=WHITE, mec=NAVY, lw=1.2, label='Landau eigenvalue')
    a2 = a.twinx()
    a2.plot(sel['mode'], sel['charge_overlap'], color=TEAL, marker='s', ms=2.8, lw=1.0, ls='--', label='charge overlap')
    a2.plot(sel['mode'], sel['axial_overlap'], color=VERMILION, marker='^', ms=2.8, lw=1.0, ls=(0,(2,2)), label='axial overlap')
    a.axhline(0, color=MID_GREY, lw=0.8)
    a.set(xlabel='patch-mode index', ylabel='dimensionless eigenvalue'); a2.set_ylabel('mode overlap')
    label_panel(a,'c',r'$Z_i Z_j \Gamma^\omega$ patch operator')
    h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels(); a.legend(h1+h2,l1+l2, loc='best', fontsize=5.7)
    # S3d
    a = ax[1,1]; polish(a, ygrid=True)
    vals = [lsum['U1_projection_residual'], lsum['U1_eigenvalue_residual'], lsum['nonabelian_covariance_residual'], lsum['nonabelian_energy_residual']]
    names = [r'$U(1)$ tensor', r'$U(1)$ spectrum', r'$U(2)$ tensor', r'$U(2)$ energy']
    colors = [NAVY, TEAL, PLUM, VERMILION]
    a.bar(np.arange(4), vals, color=colors, edgecolor='none')
    a.set_yscale('log')
    a.set_xticks(np.arange(4), names, rotation=20, ha='right')
    a.text(0.97, 0.95, rf'two-mode leakage = {100*lsum["relative_two_mode_leakage"]:.1f}%', transform=a.transAxes, ha='right', va='top', fontsize=6.0, color=SLATE)
    a.set_ylabel('gauge-covariance residual')
    label_panel(a,'d','Band-gauge covariance')
    fig.subplots_adjust(wspace=0.42, hspace=0.42)
    savefig(fig, FULLBZ_FIG / 'figS3_ward_landau_completion', 'Ward and Landau completion')


def main():
    figS1(); figS4(); figS5(); figS6(); figS7(); figS8(); figS9(); figS2S3_fullbz()
    print('Premium supplemental and appendix figures generated.')

if __name__ == '__main__':
    main()
