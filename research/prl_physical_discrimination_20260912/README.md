# PRL physical discrimination 추가 연구

읽는 순서: [한국어 연구 과정](research_journal_ko.md) → [영문 상세 PDF](physical_discrimination_note.pdf) → [대표 수치](analysis/representative_summary.csv) → [전체 실험](production_v2/trials.csv).

본문 통합용은 [main_insert.tex](main_insert.tex), 보충 통합용은 [supplement_insert.tex](supplement_insert.tex)이다. 작은 본문 그림은 [main_frequency_bias.pdf](figures/main_frequency_bias.pdf), 자세한 두 그림은 [linewidth_discrimination.pdf](figures/linewidth_discrimination.pdf)와 [scope_controls.pdf](figures/scope_controls.pdf)이다. 현재 폴더의 추가분을 기존 PRL 원고에 자동 반영하지 않았다.

11,280개 고정 합성 표본에서 물리 모형·독립 기준·오차 반경을 전제로 하는 주파수 box를 계산했다. 대표 B=.2에서 pp 주파수 편향은 단일 극 HWHM의 약 1.18배, 최대 조건부 box 반경은 0.248배다. 단일 극의 FWHM은 2η이고 전체 손실피크의 폭과 혼동하지 않는다. 정확한 교정의 약 98.8%는 음의 주파수 국소 극에서 온다. 새 원격 모드 효과나 일반적인 해석함수 복원의 증명으로 쓰지 않는다.

관심 모형이 star 구조를 벗어나거나 bright 기준·알려진 linewidth가 변하면 해당 box의 전제가 깨진다. source에 직교한 compressed root와 microscopic bare mode도 일반적으로 다르다. coupling 부호는 이 auto-response에서 식별하지 못한다. 실제 reduced Weyl의 594개 대조는 별도 데이터이며 oscillator box를 적용하지 않았다. 창밖 후보를 포함한 최대 약 6γs는 추출 실패이고, 창안/holdout RMS<.001인 308행의 최대 full-fit 오차는 약 .769γs다.

## 안전하게 재현하기

cwd: H:/Research/weyl-collective-modes. 기존 Python의 NumPy, SciPy, Pandas, Matplotlib, PyMuPDF를 사용했다. 새 의존성을 설치하지 않았다. PDF 조판은 기존 H:/Tools/tectonic-0.17.0/tectonic.exe와 H 캐시를 사용했다.

~~~powershell
& .\scripts\paper_environment.ps1
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m unittest discover -s research\prl_physical_discrimination_20260912 -p test_discrimination.py -v
python research\prl_physical_discrimination_20260912\run_production.py --output reproduction_run_01
python research\prl_physical_discrimination_20260912\run_controls_reproducible.py --output reproduction_controls_02
~~~

각 생성기는 output 폴더가 이미 있으면 중단한다. 저장된 production, production_v2, controls, reproduction_controls_01을 지우거나 덮어쓸 필요가 없다. reproduction_controls_01은 전달 전에 실행해 네 CSV의 모든 바이트가 원본 controls와 동일함을 확인했다.

runtime 보조 scripts/paper_environment.ps1이 없는 별도 압축해제 환경에서는 TEMP/TMP/MPLCONFIGDIR를 H의 전용 임시 경로에 설정하고 PYTHONDONTWRITEBYTECODE=1을 지정하면 된다. 과학적 외부 입력은 기존 pole_zero_work/code/의 weyl_continuum_stress.py, weyl_fl_prb_solver.py, pole_zero_tomography.py 세 파일뿐이다. 새 12개 테스트는 이 세 파일 없이도 새 폴더만으로 실행된다.

## 근거와 보존

- [생산 v2 진단](production_v2/diagnostic.json): 11,280행, 조건부 box의 표본 포함 실패 0, 입력·출력 해시.
- [계산 검산](analysis/validation.json): 12개 검사, 11,280행 direct 3×3/cofactor 대조 최대 6.47×10⁻¹⁵.
- [별도 대조](controls/diagnostic.json): 구조 18행, 배경 분해 20행, 유한 절편 20행, Weyl 594행.
- [재현 바이트 대조](reproduction_controls_01/byte_comparison.json): 네 CSV 바이트 동일.
- [최종 전달 검증](delivery_validation.json): 현재 PDF·그림·본문·근거 해시와 조판 검사.
- [전체 manifest](delivery_manifest.json): 자기 자신을 제외한 파일의 SHA-256.

첫 pilot의 산술 오기, 첫 생산 후 rank safeguard 추가, 인코딩이 깨진 동결 계획 단락, 그림 주석 범위 수정과 조판 이전판은 모두 별도 기록·파일로 보존했다. 기존 논문·제출 패키지·과학 데이터는 변경하지 않았다. 높은 PRL 중요성 또는 제출 승인을 이 추가 연구만으로 확정하지 않는다.

