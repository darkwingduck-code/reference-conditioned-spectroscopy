#!/usr/bin/env python3
from pathlib import Path
import json, subprocess
import fitz
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name, passed, detail=''):
    checks.append({'name':name,'passed':bool(passed),'detail':str(detail)})

code=(ROOT/'code/generate_prl_main_figures_premium.py').read_text(encoding='utf-8')
tex=(ROOT/'manuscript_prl/main_prl_candidate.tex').read_text(encoding='utf-8')
log=(ROOT/'manuscript_prl/main_prl_candidate.log').read_text(encoding='utf-8',errors='replace')
check('hybrid branches are the dominant continuous layer', '_draw_hybrid_branch' in code and 'zorder=6' in code)
check('hybrid branches use a narrow separation stroke', 'pe.Stroke(linewidth=3.15' in code)
check('reference lines are below the hybrid spectrum', 'label=label, zorder=2' in code)
check('reference identity is carried by hollow rings', 'facecolors="none"' in code and 'zorder=8' in code)
check('reconstructed zero uses hollow diamonds', 'marker="D"' in code and 'facecolors="none"' in code)
check('caption describes the balanced hierarchy', 'drawn beneath the solid hybrid branches' in tex)
check('caption explains hollow reconstruction markers', 'hollow diamonds lie on the dotted zero-field axial guide' in tex)

hybrid=np.array([0x5c,0x40,0x5d])
for stem in ['fig1_field_intercept','fig2_pole_zero_tomography']:
    pdf=ROOT/f'fig_prl/{stem}.pdf'; png=ROOT/f'fig_prl/{stem}.png'
    check(f'{stem} PDF exists', pdf.exists() and pdf.stat().st_size>20000, pdf.stat().st_size if pdf.exists() else 'missing')
    fonts=subprocess.run(['pdffonts',str(pdf)],capture_output=True,text=True)
    check(f'{stem} has no Type-3 fonts', fonts.returncode==0 and 'Type 3' not in fonts.stdout)
    with Image.open(png).convert('RGB') as im:
        arr=np.asarray(im)
        count=int((np.max(np.abs(arr.astype(int)-hybrid),axis=2)<20).sum())
        check(f'{stem} retains a strong hybrid-color footprint', count>40000, count)
        check(f'{stem} preview remains high resolution', im.width>=3000 and im.height>=1200, f'{im.width}x{im.height}')

main=ROOT/'manuscript_prl/main_prl_candidate.pdf'
with fitz.open(main) as doc:
    check('main manuscript remains three pages', doc.page_count==3, doc.page_count)
check('LaTeX log has no layout/reference failure', not any(t in log for t in ['Overfull','undefined references','LaTeX Error','Fatal error']))

payload={'all_checks_passed':all(c['passed'] for c in checks),'number_of_checks':len(checks),'checks':checks}
(ROOT/'data_prl/v13_hybrid_priority_validation.json').write_text(json.dumps(payload,indent=2)+'\n')
lines=[f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in checks]
lines.append(f"TOTAL: {len(checks)}; all_passed={payload['all_checks_passed']}")
(ROOT/'data_prl/v13_hybrid_priority_validation.log').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
raise SystemExit(0 if payload['all_checks_passed'] else 1)
