# 생산 전 고정 계획

2026-09-12. Pilot은 별도 보존합니다. pilot_plan.md의 총계 1080은 산술 오기였으며 실제 행렬은 4×3×5×2×3=360입니다. 실제 pilot.csv와 진단은 360행으로 일치합니다.

Pilot 후 선택: 대표 h=.8, q0=.7458353261772512. 최댓값 h=1.2를 주사례로 고르지 않았습니다. 두 국소 극이 들어오는 고정 관측창 [0.79,1.15]을 사용합니다. 창 중심 .97은 pilot 이후 고정 설계값이며 추정기에 참 dark-frequency나 finite-B root를 주지 않습니다. 신호와 B=0 기준의 주파수는 같고 known common broadening η를 사용합니다. 미지 linewidth 추정 문제는 이 범위에 포함하지 않습니다. η는 half width, FWHM=2η입니다.

주 생산: q0−.04,q0,q0+.04; B=.025,.05,.1,.15,.2; η=.003,.01; 신호·B0 독립 측정오차 radius=1e−5,1e−4,1e−3 × 각 raw spectral RMS; instrument calibration radius=0,1e−4,1e−3 × known-standard RMS; 각 40 bounded-complex-noise replicas. 총 10800행.

Held-out: q0±.02; B=.075,.125,.175; η=.003,.01; signal/reference radius=1e−4,1e−3; calibration radius=1e−4; 각 20 replicas. 총 480행. 생산 전체 11280행. Seed와 참조/신호 seed를 매 행 저장하며 각 q/η/noise/calibration/replica의 detector calibration과 B0 reference는 B sweep에서 공유합니다. 신호 noise는 B별 독립입니다. 추가 최적화를 하지 않습니다.

Frequency train=181 evenly spaced, synthetic clean holdout=180 interlaced frequencies. 학습 응답으로 [3/2] complex rational fit(두 극+affine background)을 수행하여 pp zero와 full-numerator zero를 각각 기록합니다. 별도로 전체 Hamiltonian의 정확한 두 양의 극·유수로 구한 pp zero를 기록합니다. 이 두 pp 추정은 같다고 가정하지 않습니다.

독립 known-standard χstd와 blank spectrum으로 M(z), additive instrument baseline을 보정합니다. 이는 실제 장비 검증이 아니라 보정 시료 응답과 오차 반경이 주어진 합성 측정 모형입니다. Detector M은 안정한 rational transfer이며 대상 창에서 nonzero입니다. 그 자체는 비영 multiplicative factor이므로 full zero를 보존하나 pp zero는 일반적으로 보존하지 않습니다. 별도 B=0 spectrum은 bright+remote block A=χ0⁻¹를 정합니다. 숨은 d,g는 calibration input에 쓰이지 않고 T=χ/(Aχ−1)의 affine coefficients에서 추정합니다.

조건부 deterministic box: 보정의 pointwise complex error radii를 역수·분수에 부등식으로 전파합니다. χ0 역수 및 Aχ−1의 양의 margin을 만족하는 data-only sample만 사용하고 제외 개수를 기록합니다. T=a(ω+iη)²+b의 real weighted least squares에 선형 영향행렬의 절댓값 bound를 적용합니다. a>0,b<0 box에서 d=sqrt(−b/a)를 interval propagation합니다. 이 bound는 exact star model, bright block의 B-independent 가정, common known η, 독립 주어진 측정오차 반경 조건에 종속됩니다. Generic unknown background, confidence interval, finite-sample analytic continuation certificate, continuum/material prediction이 아닙니다.

사전 판정: full/ref frequency error, bound width, pp bias를 η 및 2η로 보고합니다. 주 물리 대조는 허위 field curvature 또는 1 HWHM 규모의 frequency error입니다. finite-intercept 대 field-created class flip을 성공 조건으로 삼지 않습니다. 좋은 적합 heuristic은 holdout RMS≤max(5 noise, .001), fitted poles inside/lower-half-plane, condition<1e10입니다. 이 규칙 자체는 certificate가 아닙니다. 모든 탈락/상한 부재/가정 위반 negative controls를 보존합니다.


## ? ?? ? ?? ??? ?? ?? ??

? production 11280?? ?? ?? ??????. Root ?? ??? pseudoinverse? rank? ???? ?????design=I ??? ?? ? ??? ??????. production_v2? ?? seed/??/?? ??? rank=2, condition?1e10, smallest singular value, ||L X?I||?<1e?8? ???? ?? defect? ?? allowance? ?????. 5e?12 ??? arithmetic padding? ?? ??? ??? ??? ??? ????. ??? ? ??? box? ?? ??? ???? ???? ???? ?? ??? ??? ??? bound??, outward-rounded interval arithmetic certification??? ??? ????. ?? ?? ?? 5?? production/source_snapshot/? ??????.
