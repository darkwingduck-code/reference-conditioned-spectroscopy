# PRL 물리적 판별 연구: 사전 pilot 계획

이 문서는 pilot 실행 전에 기록한 설계입니다. 2026-09-12. 기존 과학 파일은 읽기만 합니다.

양의 3×3 stiffness K의 Hamiltonian H=(pᵀp+xᵀKx)/2와 좌표 응답 χcc(z)=e_cᵀ(z²I−K)⁻¹e_c를 씁니다. 모든 양·음 주파수 극을 포함합니다. Im z>0에서 −Im[zχ]≥0, 양의 실주파수에서 −Im χ≥0입니다. z=ω+iη는 유한 분해능/균일 broadening comparator이며 microscopic collision theory가 아닙니다. B는 명시된 결합을 조절하는 차원 없는 외부 제어변수이며 실제 재료 자기장 단위를 배정하지 않습니다. S=diag(1,−1,1)에 의한 K(−B)=S K(B)S는 내부 모드 parity입니다. 이것만으로 실제 재료의 시간반전 표현이나 Onsager 조건을 증명하지 않습니다.

Kcc=1+(.3q)², Kdd=(1.3q)², Krr=2.6², Kcd=.8B, Kcr=h, Kdr=0. source/readout=e_c. 이 별 구조에서는 compressed block root가 d=1.3q와 정확히 같지만 일반 source 혼합/비영 Kdr에서는 그렇지 않습니다.

pilot 행렬: h=0,.4,.8,1.2; q=q0(h)−.08,q0,q0+.08; B=.025,.05,.1,.15,.2; η=.003,.01; 주파수 반창 .10,.18,.28; 241 train와 240 interlaced holdout. 총 1080 사례. q0는 B=0에서 remote가 재규격화한 bright 모드와 dark 기준의 교차이며 식으로 독립 계산합니다. 본 실험의 잡음/오차 기준은 pilot 이후 별도 고정 설계 파일에 기록합니다.

검증할 질문: (1) 정확한 두 양의 극 유수의 pp zero가 B에 따른 가짜 dark-frequency 곡률을 주는가? (2) 충분히 좋은 실제 유한창 두 극+affine fit에서도 그 차이가 남는가? (3) full numerator 추출이 그 차이를 잡음/독립 보정오차보다 작게 줄이는가? (4) 이런 효과가 finite-intercept 대 field-created 분류까지 바꾼다는 더 강한 주장은 성립하는가? 마지막 질문의 성공을 전제하지 않습니다.

아직 사전 결정하지 않은 항목은 생산 replica 수, source transfer 함수, calibration 오차 크기, 조건부 복소 root certificate의 반지름입니다. pilot 결과로 선택한 부분은 사후 선택임을 명시하고, 다음 생산에서는 고정하며 추가 holdout을 사용합니다.

