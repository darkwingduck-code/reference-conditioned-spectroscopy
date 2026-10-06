"""Passive hidden-bath extension; existing single-mode compatibility is reused."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'research/reference_validation_20260927'))
from scipy.optimize import linprog
from experiment import Q0, OMEGA, reference_lp, verify_witness, transformed_data, LP_OPTIONS

MODELS = [('isolated', (1.1, 1.35), 0.),
          ('near_weak', (1.1, 1.35), .5), ('near_strong', (1.1, 1.35), 1.),
          ('far_weak', (1.6, 2.), .5), ('far_strong', (1.6, 2.), 1.)]
DARK = 1.3 * Q0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrices(field, bath, scale):
    k = np.diag([1 + (.3*Q0)**2, DARK**2, 2.6**2, bath[0]**2, bath[1]**2])
    k[0, 1] = k[1, 0] = .8 * field
    k[0, 2] = k[2, 0] = .8
    k[1, 3:] = k[3:, 1] = scale * np.array([.12, .08])
    gamma = np.diag([.003, .003, .007, .02, .02])
    return k, gamma


def response(z, k, gamma):
    z = np.asarray(z)
    matrix = z[..., None, None]**2*np.eye(len(k))-k+2j*z[..., None, None]*gamma
    return np.linalg.inv(matrix)[..., 0, 0]


def poles(k, gamma):
    size = len(k)
    companion = np.block([[np.zeros_like(k), np.eye(size)], [k, -2j*gamma]])
    return np.linalg.eigvals(companion)


def effective(z, bath, scale):
    z = np.asarray(z, complex)
    links = scale * np.array([.12, .08])
    return (z*z-DARK**2+2j*.003*z
            - np.sum(links**2/(z[..., None]**2-np.array(bath)**2+2j*.02*z[..., None]), axis=-1))


def rotate_bath(k, gamma):
    vector = k[1, 3:]
    norm = np.linalg.norm(vector)
    rotation = np.eye(5)
    if norm:
        a, b = vector / norm
        rotation[3:, 3:] = [[a, b], [-b, a]]
    return rotation @ k @ rotation.T, rotation @ gamma @ rotation.T


def identity_checks():
    records = []
    probes = np.r_[OMEGA, OMEGA[::9]-.015j]
    for name, bath, scale in MODELS:
        for field in (.05, .1, .2):
            k, gamma = matrices(field, bath, scale)
            k0 = k.copy(); k0[0, 1] = k0[1, 0] = 0
            x, y = response(probes, k, gamma), response(probes, k0, gamma)
            exact = effective(probes, bath, scale) / (.8*field)**2
            transformed = x*y/(x-y)
            ck, cg = rotate_bath(k, gamma)
            hidden = [1, 3, 4] if scale else [1]
            hk, hg = k[np.ix_(hidden, hidden)], gamma[np.ix_(hidden, hidden)]
            roots = poles(hk, hg)
            positive = sorted((z for z in roots if z.real > 0), key=lambda z: z.real)
            zero_residual = max(abs(response(np.array(positive), k, gamma)))
            root_residuals, full_minima = [], []
            for z in positive:
                compressed = z*z*np.eye(len(hk))-hk+2j*z*hg
                root_residuals.append(np.linalg.svd(compressed, compute_uv=False)[-1]
                    / (abs(z)**2+np.linalg.norm(hk, 2)+2*abs(z)*np.linalg.norm(hg, 2)))
                full_minima.append(np.linalg.svd(z*z*np.eye(5)-k+2j*z*gamma, compute_uv=False)[-1])
            real_response = response(OMEGA, k, gamma)
            row = dict(model=name, field=field,
                schur_scaled_error=float(np.max(abs(transformed-exact)/(1+abs(exact)))),
                star_chain_scaled_error=float(np.max(abs(x-response(probes, ck, cg))/(1+abs(x)))),
                minimum_stiffness=float(np.linalg.eigvalsh(k).min()),
                maximum_pole_imaginary=float(poles(k, gamma).imag.max()),
                minimum_loss=float((-real_response.imag).min()),
                compressed_zero_absolute_residual=float(zero_residual),
                compressed_root_backward_error=float(max(root_residuals)),
                full_kernel_minimum_singular_at_zero=float(min(full_minima)),
                hidden_positive_poles=[[float(z.real), float(z.imag)] for z in positive],
                nearest_hidden_shift=float(min(positive, key=lambda z: abs(z.real-DARK)).real-DARK),
                chain_dark_to_last_bath=float(ck[1, 4]),
                chain_bath_link=float(ck[3, 4]))
            assert row['schur_scaled_error'] < 1e-9, row
            assert row['star_chain_scaled_error'] < 1e-10, row
            assert row['minimum_stiffness'] > 0 and row['maximum_pole_imaginary'] < 0, row
            assert row['minimum_loss'] >= 0, row
            assert max(root_residuals) < 1e-12 and min(full_minima) > 1e-12, row
            records.append(row)
    bare = OMEGA**2-DARK**2+2j*.003*OMEGA
    np.testing.assert_allclose(effective(OMEGA, (1.1, 1.35), 0), bare)
    return records


def disk(rng, radius, count):
    return radius*np.sqrt(rng.random(count))*np.exp(2j*np.pi*rng.random(count))


def remainder_intervals(x, y, radius):
    """Necessary intervals given an independently supplied gapped-bath prior."""
    mask, tr, error = transformed_data(x, np.full(len(x), radius), y, np.full(len(y), radius))
    if mask.sum() < 12:
        return dict(remainder_status='inconclusive')
    w = OMEGA[mask]; h = .0208 / (1.6**2-w*w); zero = np.zeros(len(w))
    a = np.vstack([np.column_stack([np.ones(len(w)), tr.real-error, zero]),
                   np.column_stack([-np.ones(len(w)), -tr.real-error, zero]),
                   np.column_stack([zero, -tr.imag-error, 2*w]),
                   np.column_stack([zero, tr.imag-error, -2*w])])
    b = np.r_[w*w+h, -w*w+h, h, h]
    scale = np.maximum(1, np.maximum(np.max(abs(a), axis=1), abs(b)))
    a = a/scale[:, None]; b = b/scale+1e-9
    intervals = []
    for axis in np.eye(3):
        lo = linprog(axis, A_ub=a, b_ub=b, bounds=(0, None), method='highs-ds', options=LP_OPTIONS)
        hi = linprog(-axis, A_ub=a, b_ub=b, bounds=(0, None), method='highs-ds', options=LP_OPTIONS)
        if not lo.success or not hi.success:
            return dict(remainder_status='inconclusive')
        if max(np.max(a@lo.x-b), np.max(a@hi.x-b)) > 5e-9:
            return dict(remainder_status='inconclusive')
        padding = 5e-8*(1+max(abs(lo.fun), abs(hi.fun)))
        intervals.append((max(0., lo.fun-padding), -hi.fun+padding))
    if intervals[1][0] <= 0:
        return dict(remainder_status='inconclusive')
    r, t, g = intervals
    return dict(remainder_status='compatible', remainder_omega_lower=float(np.sqrt(r[0])),
                remainder_omega_upper=float(np.sqrt(r[1])), remainder_gamma_lower=float(g[0]),
                remainder_gamma_upper=float(g[1]), remainder_coupling_squared_lower=float(t[0]),
                remainder_coupling_squared_upper=float(t[1]),
                remainder_bare_frequency_in_interval=bool(r[0] <= DARK**2 <= r[1]))


def main(output):
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    checks = identity_checks()
    sources = [Path(__file__), HERE/'plan.md', REPO/'research/reference_validation_20260927/experiment.py',
               REPO/'research/physical_discrimination_20260912/oscillator_pilot.py',
               REPO/'research/physical_discrimination_20260912/discrimination_core.py']
    hashes = {p.relative_to(REPO).as_posix(): sha(p) for p in sources}
    k0, gamma = matrices(0, (1.1, 1.35), 0)
    reference = response(OMEGA, k0, gamma)
    scale0 = float(np.sqrt(np.mean(abs(reference)**2)))
    rows, witnesses, spectra = [], [], {}
    for mi, (name, bath, scale) in enumerate(MODELS):
        for bi, field in enumerate((.05, .1, .2)):
            k, gamma = matrices(field, bath, scale)
            signal = response(OMEGA, k, gamma)
            if field == .1:
                spectra[name] = signal
            for ei, epsilon in enumerate((0, 1e-4, 1e-3)):
                for rep in range(1 if epsilon == 0 else 5):
                    seed = 2026092700 + 10000*mi + 1000*bi + 100*ei + rep
                    rng = np.random.default_rng(seed)
                    radius = epsilon*scale0
                    x = signal + disk(rng, radius, len(OMEGA))
                    y = reference + disk(rng, radius, len(OMEGA))
                    result = reference_lp(OMEGA, x, np.full(len(x), radius), y, np.full(len(y), radius))
                    witness = result.pop('witness', None)
                    if witness is not None:
                        assert verify_witness(witness)
                        witnesses.append(dict(case=len(rows), witness=witness))
                    row = dict(case=len(rows), model=name, field=field, epsilon=epsilon,
                               error_radius=radius, replicate=rep, seed=seed, **result)
                    if result['status'] == 'compatible':
                        row['bare_frequency_in_interval'] = bool(result['omega_lower'] <= DARK <= result['omega_upper'])
                    if name.startswith('far_'):
                        row.update(remainder_intervals(x, y, radius))
                    rows.append(row)
    assert len(rows) == 165
    assert all(sha(REPO/name) == value for name, value in hashes.items())
    groups = []
    for name, _, _ in MODELS:
        for epsilon in (0, 1e-4, 1e-3):
            selected = [r for r in rows if r['model'] == name and r['epsilon'] == epsilon]
            groups.append(dict(model=name, epsilon=epsilon,
                **{status: sum(r['status'] == status for r in selected)
                   for status in ('compatible', 'inconsistent', 'inconclusive')},
                compatible_bare_exclusions=sum(r.get('bare_frequency_in_interval') is False for r in selected)))
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'; snapshot.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshot/path.name)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (output/'trials.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)
    report = dict(status='PASS', case_count=len(rows), reference_rms=scale0, bare_dark_frequency=DARK,
                  source_sha256=hashes, identity_checks=checks, groups=groups,
                  remainder_cases=sum('remainder_status' in r for r in rows),
                  remainder_compatible=sum(r.get('remainder_status') == 'compatible' for r in rows),
                  remainder_bare_inclusions=sum(r.get('remainder_bare_frequency_in_interval') is True for r in rows),
                  exact_stored_lp_witness_count=len(witnesses),
                  scope='Finite passive hidden bath; no microscopic collision theory or topology identification')
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (output/'witnesses.json').write_text(json.dumps(witnesses, indent=2)+'\n', encoding='utf-8')
    np.savez(output/'spectra.npz', omega=OMEGA, reference=reference, **spectra)
    render(output)
    print(json.dumps({'cases': len(rows), 'groups': groups}, indent=2), flush=True)


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)
    for name, bath, scale in MODELS:
        value = effective(OMEGA, bath, scale)
        axes[0].plot(OMEGA, value.real, label=name.replace('_', ' '))
        axes[1].plot(OMEGA, value.imag/(2*OMEGA))
    axes[0].axhline(0, c='.6', lw=.6)
    axes[0].set(xlabel='Frequency', ylabel='Re $D_{\\mathrm{eff}}$', title='(a) Hidden self-energy')
    axes[0].legend(fontsize=6.5)
    axes[1].set(xlabel='Frequency', ylabel='Im $D_{\\mathrm{eff}}/(2\\omega)$', title='(b) Apparent damping')
    summary = json.loads((output/'summary.json').read_text())
    labels = ['isolated', 'near weak', 'near strong', 'far weak', 'far strong']
    x = np.arange(5); bottom = np.zeros(5)
    for status, color in [('inconsistent', '#4c78a8'), ('compatible', '#f2b447'), ('inconclusive', '#888888')]:
        values = [sum(g[status] for g in summary['groups'] if g['model']==m[0]) for m in MODELS]
        axes[2].bar(x, values, bottom=bottom, label=status, color=color); bottom += values
    axes[2].set(xticks=x, xticklabels=labels, ylim=(0, 45), ylabel='Retained cases', title='(c) Single-mode compatibility')
    axes[2].tick_params(axis='x', labelrotation=40)
    axes[2].legend(fontsize=6.5, loc='upper right')
    fig.savefig(output/'structured_sector.pdf'); fig.savefig(output/'structured_sector.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps({'self_check': 'PASS', 'models_and_fields': len(identity_checks())}))
    elif args.output is None:
        parser.error('--output is required unless --self-check is used')
    else:
        main(args.output.resolve())
