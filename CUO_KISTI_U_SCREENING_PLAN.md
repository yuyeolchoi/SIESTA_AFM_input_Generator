# CuO AFM Hubbard U 스크리닝 작업 계획서

작성 기준일: 2026-08-04
대상 저장소 기준 커밋: `a09d2f3`
목표: 검증된 CuO bulk AFM 초기 자기배치를 유지한 채 Cu의 Hubbard `U_eff`만 변화시켜
KISTI에서 재현 가능한 SIESTA 계산을 수행하고, 전자·자기 관측량을 기준으로 후보 범위를
좁힌다.

구현 상태(2026-08-04): 저장소 측 준비 기능은 완료했다. `manual-spins`,
`prepare-u-sweep`, Nurion PBS/Neuron SLURM 배열 템플릿, U manifest 결합, `.EIG` 기반
표본 gap 분석, GUI의 exact-spin 입력과 U/gap 결과 열이 구현되어 있다. 각 U 폴더에
`--copy-file`로 전달한 pseudopotential·top-level 지원 파일을 복사하고 hash도 기록한다.
다만 실제 production 입력, pseudopotential, KISTI 계정/실행 정보와 실출력은 저장소에
없으므로 U=6 pilot 및 실출력 fixture 검증은 아직 외부 실행 단계로 남아 있다.

## 1. 핵심 결정

이번 스크리닝에서는 새 AFM 배열을 자동 추정하지 않는다. workstation에서 이미 계산이
완료된 **U=6 eV CuO bulk 입력의 원자 순서와 `DM.InitSpin`을 기준 상태로 고정**한다.
각 계산에서 바꾸는 주 독립변수는 Cu의 `U_eff`뿐이다.

서로 다른 U 계산의 총에너지를 단순히 비교하여 가장 낮은 U를 선택하면 안 된다. U가
달라지면 에너지 함수 자체가 달라지므로, U 선택은 band gap, 최종 Cu local moment,
AFM 부호 유지, 절연 상태, 구조 및 필요한 경우 동일 U 안에서의 AFM-FM 에너지 차이를
종합해서 판단한다.

초기 입력 생성과 결과 정리는 Python 3.10+ workstation에서 수행하고, KISTI에는 완성된
FDF·pseudopotential·job script만 전송한다. 로그인 노드의 Python 3.6.8은 SIESTA 실행에
필요하지 않다.

## 2. 첨부 대화와 작업 시작 전 코드의 차이

기준 커밋 `a09d2f3`을 확인했을 때 다음 제약이 있었다. 위 구현 상태에 적힌 항목은 이번
작업에서 해소되었다.

- CLI에는 아직 `manual-spins` method와 `--spin-values-file` 옵션이 없다.
- `--site-moment-file`은 `atom_index,moment` 형식의 **크기**만 읽고 내부에서 절댓값을
  취하므로 AFM 부호 파일을 대신할 수 없다.
- `manual-groups`는 알려진 up/down 원자 인덱스를 재현할 수 있지만, 검증된 입력의
  signed moment를 그대로 복원하는 기능은 아니다.
- 현재 `prepare-array`는 서로 다른 초기 자기배치를 같은 베이스 FDF에 패치하는 기능이며
  Hubbard U를 sweep하지 않는다. README도 U마다 별도의 베이스 입력/배열을 만들도록
  명시한다.
- `collect-results`는 에너지, 최종 순스핀, local spin, 부호 유지율과 수렴 여부를
  수집하지만 U 값과 band gap을 결과표에 포함하지 않는다.
- `examples/CuO_bulk.cif`는 헤더에 명시된 문헌 유사 비완화 데모 구조다. 실제 screening
  구조로 사용하지 않는다.
- `pyproject.toml`은 0.3.0이지만 `siesta_afm.__version__`은 0.2.5로 남아 있다. 작업
  provenance에는 패키지 버전 문자열 대신 Git 커밋 해시도 반드시 기록한다.

첫 실제 계산은 생성된 U=6 단일 작업으로 검증한다. pilot이 통과하기 전에는 전체 배열을
KISTI에 제출하지 않는다.

## 3. 작업 시작 전에 필요한 자료

다음 자료가 확보되어야 실제 입력 생성에 들어갈 수 있다.

1. workstation에서 정상 수렴한 U=6 CuO bulk의 최상위 FDF와 모든 `%include` 파일
2. 그 계산에 사용한 Cu/O pseudopotential 파일과 basis 관련 사용자 파일
3. 최종 구조 파일(FDF, XV 또는 CIF)과 실제 계산 당시의 원자 순서
4. U=6 계산의 SIESTA 출력과 atomic Mulliken/Hirshfeld spin 표
5. 사용한 SIESTA 버전, XC functional, pseudopotential 출처, basis, MeshCutoff, k-grid,
   smearing, SCF mixing 및 수렴 기준
6. 사용할 KISTI 시스템 이름
   - Nurion이면 PBS/`qsub`
   - Neuron이면 SLURM/`sbatch`
7. KISTI project/account, queue 또는 partition, 사용할 SIESTA executable/module, 예상
   MPI core 수와 walltime
8. 1차 U 범위와 최종 선택 기준. 정하지 않았다면 아래의 2단계 grid를 시작안으로 쓴다.

## 4. 단계별 실행 계획

### 단계 A — 기준 계산 동결과 AFM 검증

1. U=6 production 입력을 `reference/`에 읽기 전용 기준본으로 복사한다.
2. 모든 include를 펼쳐 실제로 적용되는 구조, species label, DFT+U block,
   `DM.InitSpin` 및 spin control을 확인한다.
3. `DM.InitSpin`에서 Cu 원자별 `(atom_index, signed_spin)`을 추출한다.
4. 다음 불변식을 검사한다.
   - FDF의 `NumberOfAtoms`와 좌표 행 수가 일치한다.
   - 모든 Cu가 정확히 한 번씩 spin을 가진다.
   - O 또는 다른 비자기 원소에 의도하지 않은 nonzero spin이 없다.
   - up/down 수, 초기 net spin 및 Cu 인덱스가 U=6 원본과 일치한다.
   - 원자 순서가 바뀌지 않았다.
5. Cu4O4 cell이 실제 AF1 magnetic periodicity를 표현하는지 Cu-O-Cu chain 관계와
   기준 spin 배열로 확인한다. 확인되지 않으면 결과 명칭을 “CuO AFM proxy”로 제한하고
   experimental AF1을 완전히 재현했다는 표현은 사용하지 않는다.

완료 기준: 원본 U=6의 `DM.InitSpin`을 byte-level 또는 파싱 결과 기준으로 재현하며,
원자·부호 누락이 0개다.

### 단계 B — KISTI 단일 U=6 재현 pilot

전체 sweep 전에 U=6 계산 하나만 KISTI에서 실행한다.

1. 실제 server에서 아래 환경 정보를 기록한다.

   ```bash
   hostname
   command -v qsub; command -v sbatch
   module -t avail 2>&1 | grep -i siesta
   command -v siesta
   python3 --version
   ```

2. 사용할 SIESTA module/executable과 MPI launcher의 버전을 기록한다.
3. pseudopotential과 모든 입력 파일의 SHA-256을 workstation과 KISTI 양쪽에서 비교한다.
4. `/scratch/$USER/...` 아래에 단일 작업을 만들고 scheduler로 제출한다.
5. SCF 수렴, 최종 net spin, Cu local moment, AFM sign retention, band gap과 종료 상태를
   확인한다.
6. 기존 workstation U=6 결과와 차이가 있으면 전체 sweep을 시작하지 않고 SIESTA build,
   pseudopotential, MPI, FDF include 순서 및 기본값 차이를 먼저 해결한다.

완료 기준: U=6이 KISTI에서 수렴하고 AFM 부호가 유지되며, workstation 결과와의 차이가
사전에 정한 수치 허용범위 안에 든다. 허용범위는 동일 입력의 반복 계산 변동을 보고
확정하며 임의의 숫자로 정하지 않는다.

### 단계 C — U grid 설계

첫 권장 grid는 다음과 같다.

- coarse fixed-geometry: `U_eff = 0, 2, 4, 6, 8 eV`
- fine fixed-geometry: coarse 결과의 유망 구간에서 0.5 eV 간격
- 확인 계산: 최종 후보 2~3개에 대해 더 엄격한 k-grid/MeshCutoff와 SCF 기준
- 선택 사항: 최종 후보에서만 동일 U의 FM 또는 다른 검증된 magnetic state를 계산하여
  `ΔE_AFM-FM`을 비교

모든 U는 독립적인 cold start로 실행하고 같은 `DM.InitSpin`을 사용한다. 이전 U의 DM을
다음 U에 넘기는 continuation 계산은 수렴 경로 의존성을 검사하는 보조 실험으로만 두며,
주 screening 데이터와 섞지 않는다.

1차 fixed-geometry screening에서는 원자 좌표와 cell을 고정해 U 효과를 분리한다. 구조
완화가 연구 목표에 필요하면 최종 후보에서만 같은 조건으로 별도 수행하고, fixed-geometry
결과와 relaxed 결과를 다른 표로 관리한다.

### 단계 D — 입력 생성 자동화

#### D-1. 즉시 가능한 최소 경로

1. 검증된 U=6 FDF를 각 U 디렉터리의 기준으로 복사한다.
2. 같은 `DM.InitSpin`을 현재 `siesta-afm patch`로 다시 적용해 spin block을 정규화한다.
3. DFT+U block에서는 Cu species의 U 값만 바꾼다.
4. 생성 후 기준 FDF와 diff를 검사해 허용된 차이가 다음뿐인지 확인한다.
   - U 값
   - `SystemLabel` 또는 출력 파일 label
   - provenance 주석
5. U 값 변경을 단순 전역 문자열 치환으로 처리하지 않는다. species label이 여러 개이거나
   U와 유사한 숫자가 다른 설정에 있을 수 있으므로 FDF block을 구조적으로 파싱한다.

#### D-2. 저장소 보강 작업

다음 저장소 보강은 구현되었다. 실제 KISTI 출력 fixture에 대한 회귀 검증만 외부 자료를
받은 뒤 추가한다.

1. `manual-spins` method 추가
   - CSV 스키마: `atom_index,spin`과 선택적 `element`
   - signed 값을 절댓값 변환 없이 보존
   - 선택된 모든 magnetic atom이 정확히 한 번 있어야 함
   - 누락, 중복, 범위 밖, 원소 불일치 시 중단
   - `--moment`와 동시 사용 금지
2. `prepare-u-sweep` 명령 추가
   - 입력: 검증된 base FDF, element/species label, U 값 목록, reference spin file
   - U마다 독립 디렉터리와 `RUN.fdf` 생성
   - `u_sweep_manifest.csv`에 U, 경로, 입력 hash, spin hash, Git commit 기록
   - `--copy-file` 지원 파일을 각 폴더에 복사하고 이름·SHA-256 기록
   - 기존 `prepare-array`와 의미를 섞지 않음
3. 결과 수집 확장
   - manifest의 `u_eff_ev`를 결과에 join
   - 표준 `.EIG` 파일의 SCF k-mesh gap parser 추가; 실제 KISTI fixture 검증은 대기
   - Cu별 최종 spin 평균/최솟값, sign retention, collapsed Cu 수를 기록
   - task별 `exit_code`와 실패 array index/folder 목록 기록
4. scheduler 템플릿 추가
   - Nurion 확인 시 PBS array 템플릿
   - Neuron 확인 시 SLURM array 템플릿
   - 동시 실행 개수 제한과 실패 job 재제출 목록 지원
5. 단위/통합 테스트
   - U 이외의 FDF 설정이 변하지 않는지 snapshot/diff 검사
   - signed spin round-trip
   - species label이 여러 개인 FDF
   - 잘못된 U block과 reference 누락 시 fail-fast
   - synthetic `.EIG`/출력 fixture 기반 결과 파싱 완료
   - 실제 KISTI 출력 fixture 기반 회귀 테스트는 production 자료 확보 후 추가

`reference-map`, basis-aware propagation vector 및 Cu-O-Cu superexchange graph는 동일 bulk
구조의 U sweep에는 필요하지 않다. 이후 bulk에서 slab/adsorbate로 AFM 부호를 전달할 때
별도 작업으로 진행한다.

### 단계 E — KISTI job array 실행

서버 종류를 확인한 뒤 하나의 scheduler만 채택한다.

- Nurion: 공식 지침의 PBS와 `qsub`, job array `-J` 사용
- Neuron: 공식 지침의 SLURM과 `sbatch`, `--comment siesta` 사용

job array index는 `u_sweep_manifest.csv`의 한 행과 정확히 대응시킨다. 각 task는 자기
디렉터리로 이동한 뒤 SIESTA를 실행하고 `siesta.out`, scheduler stdout/stderr, restart
파일과 종료 코드를 남긴다. pseudopotential은 각 폴더에 복사하거나 변경 불가능한 공용
경로를 사용한다. symlink를 쓸 경우 전송·아카이빙 후에도 링크가 유효한지 확인한다.

KISTI scratch 정책에 대비해 제출 전 입력을 workstation에 보관하고, 완료 직후 출력과
manifest를 다시 내려받는다. 실패한 task만 다시 제출하며, 수렴 조건을 자동으로 느슨하게
바꾼 결과를 정상 결과와 섞지 않는다.

### 단계 F — 결과 판정

각 U에 대해 최소 다음을 표로 만든다.

| 항목 | 사용 목적 |
|---|---|
| `u_eff_ev` | 독립변수 |
| SCF/geometry convergence | 유효 계산 필터 |
| final net spin | AFM 보상 여부 |
| Cu local spin 분포 | moment 유지·site collapse 확인 |
| sign retention | 초기 AFM topology 유지 확인 |
| band gap 및 direct/indirect 구분 | 전자구조 기준 |
| total energy per formula unit | 같은 U 안의 magnetic state 비교용 |
| `ΔE_AFM-FM` | 동일 U에서 AFM 안정성 확인 |
| 계산 hash와 SIESTA version | 재현성 |

선택 규칙은 계산 전에 정한다.

1. 미수렴, metallic collapse 또는 AFM 부호 붕괴 계산은 U 후보에서 제외한다.
2. band gap 하나만 맞춘 U를 바로 선택하지 않는다.
3. local moment는 Mulliken/Hirshfeld 정의와 basis에 의존하므로 같은 population method만
   비교한다.
4. 동일 U 내 AFM-FM 에너지, gap, moment 및 구조 오차의 균형을 본다.
5. 최종 후보는 더 엄격한 수치 설정에서 재계산하고 순위가 유지되는지 확인한다.

논문용 U를 주장하려면 경험적 observable matching뿐 아니라 pseudopotential/projector
의존성과 transferability를 논의한다. 필요하면 SIESTA의 potential-shift/linear-response
경로를 별도 연구로 수행한다.

## 5. 권장 디렉터리 구조

```text
cuo_u_screen/
├── reference/
│   ├── CuO_U6_original.fdf
│   ├── CuO_AF1_spin.fdf
│   ├── structure.xv
│   ├── pseudopotentials/
│   └── provenance.json
├── coarse/
│   ├── u_sweep_manifest.csv
│   ├── U_0.0/
│   ├── U_2.0/
│   ├── U_4.0/
│   ├── U_6.0/
│   └── U_8.0/
├── fine/
├── scheduler/
│   └── run_u_array.pbs 또는 run_u_array.slurm
├── results/
│   ├── coarse_results.csv
│   ├── fine_results.csv
│   └── screening_summary.md
└── checksums.sha256
```

## 6. 산출물

1. 검증된 CuO AFM reference spin 파일
2. KISTI U=6 재현 보고서
3. U별 완전한 SIESTA 입력과 manifest
4. 서버에 맞춘 PBS 또는 SLURM job array script
5. 실제 출력 fixture와 확장된 결과 CSV
6. gap, moment, sign retention, 동일 U의 magnetic energy 차이를 정리한 최종 보고서
7. 재현 명령, Git commit, SIESTA build 및 파일 hash가 포함된 README

## 7. 중단 조건

다음 중 하나가 발생하면 대량 제출 전에 멈춘다.

- 실제 U=6 production FDF 또는 pseudopotential이 없음
- reference와 target의 원자 순서가 다름
- Cu4O4 cell의 AFM 배열이 의도한 magnetic periodicity를 표현하는지 확인할 수 없음
- KISTI U=6 pilot에서 spin collapse 또는 workstation과 큰 불일치 발생
- SIESTA의 U projector 문법/버전이 기존 입력과 다름
- U 외에 k-grid, basis, XC, pseudopotential 또는 구조가 U별로 달라짐

## 8. 참고 자료

- [SIESTA 5.4 DFT+U/DFTU.Proj reference](https://docs.siesta-project.org/projects/siesta/en/5.4/reference/siesta.html)
- [SIESTA DFT+U tutorial](https://docs.siesta-project.org/projects/siesta/en/school-2021/tutorials/dft%2Bu/index.html)
- [Dudarev et al., rotationally invariant DFT+U](https://doi.org/10.1103/PhysRevB.57.1505)
- [CuO AF1 magnetic structures from single-crystal neutron diffraction](https://www.nature.com/articles/ncomms10295)
- [CuO experimental/theoretical electronic structures and optical gap](https://doi.org/10.1103/PhysRevB.94.245418)
- [KISTI Nurion PBS job-script keywords](https://docs-ksc.gitbook.io/nurion-user-guide-eng/appendix/appendix-1-main-keywords-for-job-scripts)
- [KISTI Neuron SLURM execution guide](https://docs-ksc.gitbook.io/neuron-user-guide-eng/system/running-jobs-through-scheduler-slurm)
