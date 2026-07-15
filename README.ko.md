# siesta-afm

**[English](README.md)** · 한국어

`siesta-afm`은 CIF, XYZ, POSCAR/CONTCAR, SIESTA XV 및 FDF 구조에서 자기성 원자를 선택하고 SIESTA용 `%block DM.InitSpin` 초기 스핀 배열을 만드는 Python CLI입니다. 입력 원자 순서를 바꾸지 않으며 SIESTA 출력 인덱스는 항상 1부터 시작합니다.

## 설치

Python 3.10 이상이 필요합니다.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
```

산화수 추정, 시각화, GUI는 각각 선택 의존성입니다.

```bash
python -m pip install -e ".[oxidation,plot,gui,yaml]"
```

## 빠른 시작

```bash
siesta-afm analyze examples/CuO_111_slab.cif \
  --magnetic-species Cu --slab

siesta-afm generate examples/CuO_111_slab.cif \
  --magnetic-species Cu \
  --method layer \
  --axis z \
  --moment 0.5 \
  --slab \
  --output afm_spin.fdf

siesta-afm validate afm_spin.fdf \
  --structure examples/CuO_111_slab.cif \
  --magnetic-species Cu --slab

siesta-afm plot examples/CuO_111_slab.cif \
  --spin-file afm_spin.fdf \
  --output afm_pattern.png --slab
```

`plot`은 기본적으로 스핀 부호를 빨강/파랑으로 표시합니다. 서로 다른 site moment를
연속 색으로 비교하려면 `--color-mode value`를 사용합니다. 이 모드에서는 0을 중심으로
대칭인 색 범위와 `initial spin (μB)` 컬러바가 추가되며, `--up-color`와
`--down-color`는 무시됩니다.

기존 SIESTA 입력에 결과를 삽입하려면:

```bash
siesta-afm patch examples/input.fdf \
  --spin-file examples/afm_spin.fdf \
  --output input_afm.fdf
```

`--in-place`를 지정하지 않으면 원본 FDF를 덮어쓰지 않습니다. `--backup`을 함께 사용하면 원본 옆에 `.bak` 파일을 만듭니다.

## 생성 방법

- `alternating-index`: 선택된 자기성 원자 목록 안에서만 `+ - + -`를 배정합니다.
- `layer`: `--axis` 방향 좌표를 `--layer-tolerance`로 묶고 층마다 부호를 교대합니다.
- `checkerboard`: `--plane xy|xz|yz` 평면 최근접 그래프를 이분 색칠합니다.
- `neighbor-bipartite`: PBC 최소 이미지 최근접 그래프를 만들고 두 sublattice를 색칠합니다.
- `graph-coloring`: DSATUR proper coloring으로 최대 k개 부격자 후보를 만들고 색별 collinear spin을 매핑합니다.
- `propagation-vector`: `sign(cos(2π q·r + phase))`로 부호를 정합니다.
- `manual-groups`: `--up-atoms`, `--down-atoms` 또는 YAML `--group-file`을 사용합니다.
- `by-species`: 서로 다른 원소 sublattice를 `--up-species`와 `--down-species`로 나눕니다.
- `by-coordination`: 자기 원자의 첫 anion shell 배위수로 Td/Oh sublattice를 나눕니다.
- `random`: `--seed`로 재현 가능한 무작위 초기 부호를 만듭니다. 물리적 자기질서 모델은 아닙니다.

여러 자기 원소를 함께 선택하면 `layer`는 모든 원소를 합친 좌표 층을 기준으로 부호를
교대합니다. 한 원소가 결합 층의 짝수 또는 홀수 층에만 존재하면 다른 원소는 정상적으로
교대해도 그 원소 전체가 같은 부호를 받을 수 있으며, 프로그램은 이를 경고합니다.
다원소 스피넬 ferrimagnet에는 `by-coordination`을 우선 사용하고, 원소별 독립 layer
패턴이 물리적으로 의도된 경우에만 원소를 따로 처리하십시오.

예:

```bash
siesta-afm generate structure.cif \
  --magnetic-species Ni Co \
  --method neighbor-bipartite \
  --moment Ni=1.0 Co=1.5 \
  --cutoff auto \
  --output afm_spin.fdf

siesta-afm generate structure.cif \
  --magnetic-species Cu \
  --method propagation-vector \
  --q-vector 0.5 0.5 0.5 \
  --moment 0.5

siesta-afm generate structure.cif \
  --magnetic-species Cu \
  --method manual-groups \
  --up-atoms 2,5,8,11 \
  --down-atoms 3,6,9,12 \
  --moment 0.5

siesta-afm generate spinel.cif \
  --magnetic-species Fe \
  --method by-coordination \
  --anion-species O \
  --up-coordination 6 --down-coordination 4 \
  --moment Fe@6=4.0 Fe@4=3.0

siesta-afm generate examples/Co3O4_spinel_COD1538531.cif \
  --magnetic-species Co \
  --method by-coordination \
  --moment Co@4=3.0 Co@6=0.5

siesta-afm generate inverse_spinel.cif \
  --magnetic-species Ni Co \
  --method by-coordination \
  --anion-species O \
  --moment Ni@6=2.0 Co@4=2.0 Co@6=0.0
```

`--moment 0.5`는 모든 선택 원소에 같은 크기를 쓰고, `--moment Cu=0.5 Ni=1.0`은 원소별 값을 씁니다. `Element@CN=value`는 같은 원소의 서로 다른 배위 환경을 구분합니다. 적용 우선순위는 site CSV > `Element@CN` > `Element` > 전역 값입니다. `--site-moment-file moments.csv`의 CSV에는 최소 `atom_index,moment` 열이 필요하고, 선택적으로 `element,oxidation_state` 열을 둘 수 있습니다.

`--moment`와 `--moment-config`를 모두 생략하면 다음의 일반적인 고스핀 추정값을
초기값으로 사용하며, 실제 적용된 기본값은 반드시 경고로 나열됩니다.

| 원소 | μB | 원소 | μB | 원소 | μB |
| --- | ---: | --- | ---: | --- | ---: |
| Ti | 2.0 | V | 3.0 | Cr | 3.0 |
| Mn | 5.0 | Fe | 4.0 | Co | 3.0 |
| Ni | 2.0 | Cu | 1.0 | Gd | 7.0 |

이 값은 산화수와 스핀 상태를 무시한 시작 추정값일 뿐입니다. 예를 들어 저스핀
Co³⁺는 약 0 μB일 수 있습니다. 배위수별 기본값은 자동으로 만들지 않습니다. 표에 없는
원소는 명시적인 moment가 필요하며, 부분적인 `--moment` 지정도 기본값과 조용히 섞지
않고 기존처럼 오류로 처리합니다.

생성된 `DM.InitSpin` 행에는 기본적으로 원소 인라인 주석이 붙습니다.
`by-coordination`에서는 CN=4/6을 배위수 기준으로 Td/Oh라고 함께 표시합니다. 구형
후처리 스크립트와의 호환성이 필요하면 `--no-site-comments`를 사용하십시오. 이 옵션은
스핀 값에는 영향을 주지 않습니다.

`by-species`의 up/down 합집합은 `--magnetic-species`와 정확히 같아야 합니다. 이 방법은 Ni/Co처럼 원소가 다른 sublattice에는 적합하지만, 같은 원소가 Td와 Oh 자리를 모두 차지하는 inverse spinel은 구분하지 못하므로 `by-coordination`을 사용해야 합니다. `by-coordination`은 O, S, Se, Te, N, F, Cl 중 구조에 하나만 존재하면 anion을 자동 감지하며, 여러 후보가 있으면 `--anion-species`를 요구합니다. 같은 basis anion의 서로 다른 주기 이미지도 각각 별도 이웃으로 세며, 기본 분류는 up CN=6, down CN=4입니다. `--anion-cutoff`, `--coordination-tolerance`로 판정을 조정할 수 있습니다.

위 역스피넬 명령은 기본 배위 부격자 분류를 이용해 Ni(Oh)=+2 μB,
Co(Td)=−2 μB, 저스핀 Co(Oh)=0을 만듭니다. CN=4와 CN=6 Co에 하나의
`Co=value`를 사용해도 실행은 허용하지만, 두 Co 부격자를 독립적으로 나타낼 수 없으므로
이제 경고가 출력됩니다. 여기서 Co(Oh)=0은 팔면체 Co³⁺를 저스핀 반자성으로 배정한
문헌과 부합합니다.

실험적 근거는 Zhu et al.,
[“Electronic structure and magnetic properties of spinel NiCo2O4 epitaxial thin
films,” *Scientific Reports* **5**, 15201 (2015)](https://doi.org/10.1038/srep15201)입니다.
이 XAS/XMCD 연구에 따르면 NiCo₂O₄는 페리자성 역스피넬(`Fd-3m`, Curie 온도 673 K)이며,
사면체 Td(A) 자리에는 Ni 없이 고스핀 Co²⁺/Co³⁺가, 팔면체 Oh(B) 자리에는 고스핀
Ni²⁺/Ni³⁺와 저스핀 반자성 Co³⁺(`S=0`)가 섞여 있습니다. 자기·전도 특성은 단순한
2-부격자 AFM이 아니라 성장 조건에 따라 달라지는 Ni³⁺ 농도와 함께 반강자성
초교환(super-exchange)과 강자성·전도성 이중교환(double-exchange)이 경쟁한 결과입니다.

`by-coordination`은 배위수(Td/Oh)만으로 부격자를 나눕니다. 같은 배위 자리 안에 섞인
Ni²⁺/Ni³⁺ 또는 Co³⁺ HS/LS 같은 산화수·스핀 상태는 구분하지 못합니다. 정확한
산화수별 자리를 반영하려면 `--site-moment-file`로 원자별 moment를 직접 지정하거나,
산화수에 따라 구조 파일의 원자를 그룹화해 별도로 다루십시오.

Co₃O₄ 예제는 [COD 1538531](https://www.crystallography.net/cod/1538531.html)
(Roth, 1964)의 퍼블릭 도메인 구조 데이터를 사용합니다. Co 배위수 분포는 Td CN=4가 8개,
Oh CN=6이 16개입니다.

Propagation vector의 q는 입력 cell의 fractional 좌표입니다. supercell에서는 같은 물리적 주기를 나타내도록 q를 축소해야 합니다. A/C/G preset은 각각 `--afm-type A`, `C`, `G`로 선택하며, 사용자 `--q-vector`와 동시에 쓸 수 없습니다. NiO의 (111) AFM-II처럼 축과 평행하지 않은 층은 `--method layer --layer-direction 1 1 1`로 생성합니다.

YAML 설정은 `--moment-config moments.yaml`로 읽습니다.

```yaml
moments:
  Cu: 0.5
  Ni: 1.0
  Co: 1.5
```

산화수 자동 추정은 기본으로 실행하지 않습니다. 사용자가 `--guess-oxidation-states`를 명시하고 `pymatgen` 선택 의존성이 설치된 때에만 수행하며, 결과가 추정값이라는 경고를 냅니다.

## 주기경계, slab, 흡착종

입력 파일의 PBC를 쓰거나 명시적으로 바꿀 수 있습니다.

```bash
--slab                 # xy periodic, z nonperiodic
--periodic-axes xy     # 같은 의미의 명시적 설정
--periodic-axes xyz    # 3차원 주기 구조
```

자기성 원소는 반드시 `--magnetic-species`로 선택하므로 C, H, O, Cs 같은 흡착종은 자동으로 자기 그래프에서 빠집니다. 추가 제외 범위는 `--exclude-atoms 217-228` 또는 `--adsorbate-indices 217,218,219`로 지정합니다.

## Frustrated/non-bipartite 그래프

`neighbor-bipartite` 그래프가 이분 그래프가 아니면 프로그램은 임의의 2-sublattice 결과를 만들지 않고 오류로 끝납니다. 이 경우 layer, propagation vector, manual group 또는 다른 cutoff를 검토하십시오.

`--allow-frustrated`는 반대 부호 edge 수를 늘리는 반복 Max-Cut 휴리스틱을 명시적으로 허용합니다. 이 결과에는 다음 과학적 경고가 기록됩니다.

> The generated spin assignment is a heuristic initial state for a frustrated magnetic network. It is not guaranteed to represent the experimental magnetic ground state.

그래프가 둘 이상의 connected component로 끊기면 각 성분 내부의 교대 부호만 그래프로
결정됩니다. 성분 사이 상대 부호는 최소 원자 인덱스를 기준으로 한 결정론적 관례일 뿐
물리적 의미가 없습니다. 이때 프로그램은 성분 수와 크기를 경고하며, 층간 초교환을
포함하도록 `--neighbor-cutoff`를 조정하거나 `layer`/`propagation-vector` 방법을
검토하라고 안내합니다. 비주기 방향에 홀수 자기층을 가진 layer 슬랩은 오류가 아니라
비보상 AFM 슬랩이라는 정보성 경고를 냅니다.

`graph-coloring`은 비이분 그래프에 DSATUR를 적용하는 다부격자 초기 후보 생성기입니다.
`--max-colors` 기본값은 4이며, `--color-spins "+1,-1,0"`으로 색별 부호를 지정하거나
`--balance-colors`로 `--moment`·원소별 moment·site moment 파일에서 해석한 실제 초기
moment 합의 절대값이 가장 작은 색-부호 순열을 선택할 수 있습니다.
proper coloring은 인접 원자의 동색을 피할 뿐 에너지를 최소화하지 않습니다. frustrated
격자의 collinear 에너지 후보가 목적이면 `--allow-frustrated` max-cut이 더 적합합니다.
`enumerate`에서는 attempt seed에 따라 색-spin 순열을 바꿔 후보를 다양화합니다.

## 분석과 검증

`analyze`는 자기 원자 거리 shell, 자동 cutoff, graph 크기/연결성/이분성, 층 수를 출력하며 `--json analysis.json`을 지원합니다.

`validate`는 중복/범위 밖 인덱스, 선택 자기 원소가 아닌 원자의 nonzero spin, up/down 수, 순스핀을 검사합니다. `--structure`를 주면 최근접 edge의 antiparallel 비율(`AFM score`), 연결 성분 및 층별 분포도 계산합니다.

## 총에너지로 여러 자기 상태 비교하기

`make-input` + `enumerate` + `prepare-array` + `collect-results`를 조합하면 계산 조건은
고정하고 초기 자기 상태만 바꾸어 비교할 수 있습니다.

1. `make-input`을 한 번 실행해 구조, basis, pseudopotential 선언, DFT+U, k-grid, SCF
   설정이 모두 들어 있는 완전한 베이스 입력을 만듭니다. 초기 스핀은 나중에 치환되므로
   어떤 방법으로 생성해도 됩니다.
2. `enumerate --methods ...`로 서로 다른 스핀 배치 후보를 만듭니다. 출력 FDF 조각에는
   후보별 `DM.InitSpin` 블록이 들어 있습니다.
3. 완전한 베이스 입력과 후보 디렉터리를 `prepare-array`에 넘깁니다. 내부의
   `patch_fdf_text`는 각 작업 폴더의 베이스 입력에서 `Spin polarized` 설정과
   `DM.InitSpin` 블록만 바꿉니다. 따라서 구조와 나머지 모든 계산 조건은 배열 전체에서
   동일합니다.
4. 클러스터에서 각 폴더의 `RUN.fdf`로 SIESTA를 실행합니다. SIESTA 실행 자체는 이 도구의
   범위 밖입니다.
5. `collect-results`로 `results.csv`를 만듭니다. `scf_converged=True`인 행 가운데
   `total_energy`가 가장 낮은 상태를 우선 후보로 선택합니다. NiCo₂O₄처럼 경쟁하는
   준안정 상태의 에너지가 서로 가까우면 하나를 임의로 고르지 말고 최종 자기 진단값과
   전체 순위를 함께 보고하십시오.

다음은 저장소의 Co₃O₄ COD fixture를 이용한 실행 가능한 예입니다. `enumerate`의
균일한 값은 두 방법 모두에서 moment 크기를 정의합니다. `base_input.fdf`의 배위별 스핀은
자리만 채우는 초기값이며 각 작업에서 치환됩니다.

```bash
siesta-afm make-input examples/Co3O4_spinel_COD1538531.cif \
  --magnetic-species Co --method by-coordination \
  --moment Co@4=3.0 Co@6=0.0 --output base_input.fdf

siesta-afm enumerate examples/Co3O4_spinel_COD1538531.cif \
  --magnetic-species Co --moment Co=3.0 Co@4=3.0 Co@6=3.0 \
  --methods by-coordination,frustrated --n-configs 2 \
  --output-dir afm_candidates

siesta-afm prepare-array base_input.fdf \
  --configs afm_candidates --output-dir siesta_jobs

# ... 클러스터에서 각 siesta_jobs/*/RUN.fdf를 SIESTA로 실행 ...

siesta-afm collect-results siesta_jobs
```

`enumerate`는 `--methods by-coordination,layer,frustrated`처럼 위상학적으로 다른 배치를
비교하는 데 적합합니다. method와 attempt seed만 바꾸며 **moment 크기나 Hubbard U는
스윕하지 않습니다.** 동일 패턴을 제거하고 기본적으로 전체 up/down 부호 반전도 같은
후보로 봅니다. 전역 부호 반전은 시간 역전 대칭 때문에 collinear 계산에서 같은 에너지를
가지므로 보통 별도 계산할 필요가 없습니다. 두 관례가 특별히 모두 필요할 때만
`--keep-global-spin-inversion`을 사용할 수 있습니다.

산화수와 스핀 상태 자체가 불확실하다면 스핀 파일을 별도로 만드십시오. 예를 들어
Co(Oh)=0(LS)과 Co(Oh)=비0(HS)으로 `generate` 또는 `make-input`을 각각 호출합니다.
그 파일들과 `enumerate` 후보를 한 설정 디렉터리에 복사하고, 각 파일의 `config_id`와
파일명을 고유하게 정한 뒤 다음의 정확한 스키마로 `manifest.csv`를 합치거나 작성합니다.

```csv
config_id,method,n_up,n_down,net_spin,afm_score,file
```

각 행은 스핀 파일 하나를 나타냅니다. `method`에는 `by-coordination-ls`,
`by-coordination-hs` 같은 label을 쓰고, 해당 파일의 up/down 수, 순스핀, AFM score를
유지하거나 다시 계산하십시오. 그러면 `prepare-array`가 모든 후보를 같은 베이스 입력에
patch할 수 있습니다. Hubbard U는 베이스 계산 설정이므로 U마다 별도의 베이스 입력과 작업
배열을 만들어야 합니다. 원자별 산화수 배정에는 배위수에 의존하기보다
`--site-moment-file`을 사용하는 편이 안전합니다.

`collect-results`는 각 폴더의 일반적인 SIESTA `.out`/`.log` 표현에서 에너지, 최종 순스핀, 원자별 Mulliken/Hirshfeld spin, 부호 유지율, collapse 및 수렴 표지를 읽어 `results.csv`로 씁니다. SIESTA 버전에 따라 출력 문구가 다르면 정규식 확장이 필요할 수 있습니다.

## GUI 실행

먼저 GUI 선택 의존성을 설치합니다.

```bash
python -m pip install -e ".[gui]"
```

Windows에서는 저장소 루트의 `run_gui.bat`를 더블클릭하면 브라우저가 아닌 Tkinter
데스크톱 창이 열립니다. 런처는 `.venv\Scripts\python.exe`를 먼저 사용하고, 없으면
PATH의 `python`을 사용합니다. Tkinter 또는 matplotlib를 사용할 수 없으면 설치 명령을
표시한 채 창을 유지합니다. 브라우저 기반 서버는 필요하지 않습니다.

터미널에서는 다음 두 진입점이 동일한 GUI를 실행합니다.

```bash
python -m siesta_afm.gui
siesta-afm-gui
```

CIF, FDF, XYZ, POSCAR/CONTCAR 또는 XV를 열면 구조에 있는 모든 원소가 magnetization
테이블에 즉시 표시됩니다. 기본 moment 표에 있는 원소만 기본 체크되고 음이온과 미지원
원소는 체크 해제됩니다. 이 테이블이 `use | element | label | CN | value (μB) | count |
role`의 정본입니다. use 셀을 더블클릭해 원소를 선택하고 label, value 및
`by-species`의 role 셀을 더블클릭해 편집합니다. element, CN, count는 읽기 전용입니다.
테이블 아래에는 동일한 `--magnetic-species ... --moment ...` CLI 옵션이 표시되어 배치
스크립트로 그대로 옮길 수 있습니다. 원자별 지정용 `Site moment file`은 유지되며
우선순위는 CLI와 동일한 site CSV > 테이블(`Element@CN` 또는 `Element`) 값입니다.

`by-coordination`에서는 `(Element, CN)` 그룹마다 한 행을 만들고, CN만이 아니라 실제
리간드 벡터로 표시 기하를 판정합니다. 예를 들어 170° 이상인 trans 리간드 쌍을 세어
CuO의 square-planar Cu(CN=4)와 스피넬의 tetrahedral Co(CN=4)를 구분합니다. 기하
라벨은 편집 가능한 추정 표시이고 moment 문법은 계속 `Element@CN`입니다. 사용자가
수정한 라벨은 해당 `DM.InitSpin` 주석에도 반영됩니다. 배위 분석에 실패해도 테이블을
비우지 않고 원소 단위 행으로 폴백하며 상태바에 이유를 표시합니다.

선택한 method에 필요한 설정 그룹만 표시됩니다. 입력과 결과 영역은 드래그 가능한
분할창이고, 짧은 라벨과 별도 도움말을 사용해 기본 창 크기에서도 입력 위젯이 잘리지
않습니다. 파라미터 변경은 계속 400 ms 디바운스로 자동 반영됩니다. `Include element/CN
comments in DM.InitSpin` 체크박스는 CLI의 `--no-site-comments`와 같은 기능을 제어하고,
기본 moment와 스핀 상태 경고는 상태바와 `Analysis` 탭에 표시됩니다. 기존 스핀 파일도
현재 구조 위에서 열 수 있습니다.

생성 후 `Sites` 탭에는 모든 자기 원자가 입력 순서대로 element, CN, sublattice, sign,
moment와 함께 표시됩니다. 하단에서 `n_up`, `n_down`, `n_zero`, 초기 net moment를 바로
확인할 수 있습니다. 스핀 파일 뷰어에서도 같은 표를 사용하되, 스핀 블록에 저장되지 않는
CN과 sublattice는 `-`로 표시합니다.

CLI의 `analyze`에 대응하는 분석은 별도 버튼이 아니라 생성/실시간 갱신 때 자동으로
실행되며 오른쪽 `Analysis` 탭에 거리 shell, cutoff, 연결성, 이분성 및 layer 수로
표시됩니다.

Export에서는 DM.InitSpin 블록, 원본을 덮어쓰지 않는 패치된 SIESTA 입력, initial
magmom이 포함된 XYZ/CIF 구조를 저장할 수 있습니다. CLI가 기준 과학 구현이며 GUI
컨트롤러도 같은 코어 함수를 사용합니다.
완전한 실행 시작 FDF를 만들려면 `Generate / View`의 눈에 띄는
`Build complete SIESTA input (make-input)...` 버튼을 사용하십시오. Export 그룹의
동일 기능 버튼도 보조 진입점으로 유지됩니다.

## 입력과 인덱스 보존

ASE가 CIF, XYZ, POSCAR/CONTCAR를 읽습니다. XV에는 ASE 실패 시 사용하는 별도 parser가 있습니다. FDF parser는 다음 block 및 재귀 `%include`를 처리합니다.

- `ChemicalSpeciesLabel`
- `AtomicCoordinatesAndAtomicSpecies`
- `LatticeVectors`

어떤 입력에서도 원소 또는 좌표 기준 정렬을 하지 않습니다. 내부 `ase_index`는 0-based 원래 순서이며 `siesta_index`는 같은 순서의 1-based 인덱스입니다.

## 중요한 과학적 주의사항

1. `DM.InitSpin` 값은 초기 추정값일 뿐 최종 local magnetic moment가 아닙니다.
2. SCF 후 spin 배열은 바뀔 수 있습니다.
3. 두 sublattice AFM이 모든 산화물에 적합한 것은 아닙니다.
4. CuO(111), triangular surface, spinel 구조는 frustrated magnetic network가 될 수 있습니다.
5. 실험 또는 문헌의 magnetic ordering이 알려진 경우 그것을 우선 사용해야 합니다.
6. 여러 AFM/FM 초기 상태를 계산하고 최종 total energy를 비교해야 합니다.
7. U 값, basis, pseudopotential, slab termination에 따라 magnetic ground state가 달라질 수 있습니다.

`examples/`의 작은 구조는 CLI 동작 확인용이며 수렴된 표면 계산 모델을 대신하지 않습니다.
특히 `NiCo2O4_311_slab.cif`는 다원소 입출력 데모로, Td/Oh 배위가 구성된 실제
spinel 구조가 아니므로 `by-coordination` 검증 모델로 사용하면 안 됩니다.

## 완전한 SIESTA 시작 입력

`make-input`은 원자 순서를 보존한 구조, 선택한 AFM 방법, `DM.InitSpin`, basis와
pseudopotential 요구사항, PBE/SCF 설정, k-grid, 선택적 DFT+U 블록을 하나의 FDF로
생성합니다.

```bash
siesta-afm make-input inverse_spinel.cif \
  --magnetic-species Ni Co \
  --method by-coordination \
  --anion-species O \
  --moment Ni@6=2.0 Co@4=2.0 Co@6=0.0 \
  --hubbard-u Ni=6.0 Co=3.3 \
  --output input.fdf
```

좌표 행과 `DM.InitSpin` 인덱스는 입력 원자 순서와 정확히 일치합니다. FDF/XV의
species ID는 보존하고, 다른 형식은 원소가 처음 등장한 순서대로 ID를 부여합니다.
자동 k-grid는 주기축마다 `ceil(30 Ang / |a_i|)`, 비주기축은 1입니다.
`--kgrid N1 N2 N3`, `--kgrid-cutoff K`,
`--basis-size SZ|SZP|DZ|DZP|TZP`로 시작값을 바꿀 수 있습니다.

기본적으로 지원되는 자기 원소에는 Materials Project의 산화물 보정 U 값을
`LDAU.proj` 문법으로 적용합니다. `--hubbard-u Element=value`로 덮어쓰고,
`--no-lda-u`로 끄거나 `--dftu-keyword dftu`로 최신 `DFTU.Proj` 표기를 선택할 수
있습니다. collinear 설정은 Dudarev의 `U_eff = U - J`를 사용하므로 템플릿은
`U=U_eff`, `J=0`으로 기록합니다. DFT+U는 `(원소, CN)` 행이 아니라 SIESTA species
단위이므로, 같은 원소가 여러 배위 sublattice를 차지하면 경고합니다.

생성 파일은 시작 템플릿일 뿐 논문 계산에 바로 사용할 수 있는 입력이 아닙니다.
pseudopotential, basis, MeshCutoff, k-grid, SCF 설정, Hubbard U, 최종 자기 상태를
반드시 수렴·검증해야 합니다. 근거와 최신 문법은
[SIESTA pseudopotential 안내](https://siesta-project.org/siesta/Documentation/Pseudopotentials/),
[SIESTA DFT+U 매뉴얼](https://docs.siesta-project.org/projects/siesta/en/stable/reference/siesta.html),
[Materials Project U 값 방법론](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga%2Bu-calculations/hubbard-u-values)을
참고하십시오.

GUI의 `by-coordination` 테이블에서 `use`는 이제 `(원소, CN)`별로 독립적입니다.
예를 들어 `Co@6`만 해제해도 `Co@4`가 켜져 있으면 Co는 자기 원소로 남습니다.
그 상태에서 `Co@6` moment가 없으면 CLI와 같은 부분 moment 오류가 발생합니다.
해당 사이트를 0 스핀으로 두려면 moment를 `0.0`으로 설정하고, 특정 원자를 완전히
제외하려면 `--exclude-atoms` 또는 `--adsorbate-indices`를 사용하십시오. GUI Export의
`Complete SIESTA input...` 버튼은 `make-input`과 같은 렌더러와 경고를 사용합니다.

## 테스트

```bash
python -m pytest
```

테스트는 FDF/include/patch, 1D·square·triangle·분리 graph, slab PBC, layer clustering, ordering, 1-based writer 및 validation 오류 검출을 다룹니다.
