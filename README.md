# siesta-afm

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
- `propagation-vector`: `sign(cos(2π q·r + phase))`로 부호를 정합니다.
- `manual-groups`: `--up-atoms`, `--down-atoms` 또는 YAML `--group-file`을 사용합니다.
- `by-species`: 서로 다른 원소 sublattice를 `--up-species`와 `--down-species`로 나눕니다.
- `by-coordination`: 자기 원자의 첫 anion shell 배위수로 Td/Oh sublattice를 나눕니다.
- `random`: `--seed`로 재현 가능한 무작위 초기 부호를 만듭니다. 물리적 자기질서 모델은 아닙니다.

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
```

`--moment 0.5`는 모든 선택 원소에 같은 크기를 쓰고, `--moment Cu=0.5 Ni=1.0`은 원소별 값을 씁니다. `Element@CN=value`는 같은 원소의 서로 다른 배위 환경을 구분합니다. 적용 우선순위는 site CSV > `Element@CN` > `Element` > 전역 값입니다. `--site-moment-file moments.csv`의 CSV에는 최소 `atom_index,moment` 열이 필요하고, 선택적으로 `element,oxidation_state` 열을 둘 수 있습니다.

`by-species`의 up/down 합집합은 `--magnetic-species`와 정확히 같아야 합니다. 이 방법은 Ni/Co처럼 원소가 다른 sublattice에는 적합하지만, 같은 원소가 Td와 Oh 자리를 모두 차지하는 inverse spinel은 구분하지 못하므로 `by-coordination`을 사용해야 합니다. `by-coordination`은 O, S, Se, Te, N, F, Cl 중 구조에 하나만 존재하면 anion을 자동 감지하며, 여러 후보가 있으면 `--anion-species`를 요구합니다. 같은 basis anion의 서로 다른 주기 이미지도 각각 별도 이웃으로 세며, 기본 분류는 up CN=6, down CN=4입니다. `--anion-cutoff`, `--coordination-tolerance`로 판정을 조정할 수 있습니다.

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

## 분석과 검증

`analyze`는 자기 원자 거리 shell, 자동 cutoff, graph 크기/연결성/이분성, 층 수를 출력하며 `--json analysis.json`을 지원합니다.

`validate`는 중복/범위 밖 인덱스, 선택 자기 원소가 아닌 원자의 nonzero spin, up/down 수, 순스핀을 검사합니다. `--structure`를 주면 최근접 edge의 antiparallel 비율(`AFM score`), 연결 성분 및 층별 분포도 계산합니다.

## 여러 후보와 SIESTA 계산 배열

```bash
siesta-afm enumerate structure.cif \
  --magnetic-species Cu \
  --moment 0.5 \
  --methods layer,checkerboard,frustrated \
  --n-configs 8 \
  --output-dir afm_configs

siesta-afm prepare-array examples/input.fdf \
  --configs afm_configs \
  --template input_setting.fdf \
  --output-dir siesta_afm_jobs

siesta-afm collect-results siesta_afm_jobs
```

`enumerate`는 동일 패턴을 제거하고 기본적으로 전체 부호 반전도 같은 후보로 봅니다. `--keep-global-spin-inversion`으로 두 반전을 별도로 유지할 수 있습니다. `manifest.csv`에는 방법, up/down 수, 순스핀, AFM score가 기록됩니다.

`collect-results`는 각 폴더의 일반적인 SIESTA `.out`/`.log` 표현에서 에너지, 최종 순스핀, 원자별 Mulliken/Hirshfeld spin, 부호 유지율, collapse 및 수렴 표지를 읽어 `results.csv`로 씁니다. SIESTA 버전에 따라 출력 문구가 다르면 정규식 확장이 필요할 수 있습니다.

## GUI 실행

먼저 GUI 선택 의존성을 설치합니다.

```bash
python -m pip install -e ".[gui]"
```

Windows에서는 저장소 루트의 `run_gui.bat`를 더블클릭하면 브라우저가 아닌 Tkinter
데스크톱 창이 열립니다. 런처는 `.venv\Scripts\python.exe`를 먼저 사용하고, 없으면
PATH의 `python`을 사용합니다. Tkinter 또는 matplotlib를 사용할 수 없으면 설치 명령을
표시한 채 창을 유지합니다. Streamlit은 필요하지 않습니다.

터미널에서는 다음 두 진입점이 동일한 GUI를 실행합니다.

```bash
python -m siesta_afm.gui
siesta-afm-gui
```

GUI는 구조 파일 선택, 자기 원소/방법/moment/cutoff/layer 설정, species·coordination
sublattice, A/C/G preset과 임의 layer 방향, 회전·확대 가능한 3D 미리보기,
graph 분석과 DM.InitSpin 미리보기를 제공합니다. 파라미터 변경은 400 ms
디바운스로 자동 반영되며 `Live update`를 끌 수 있습니다. 기존 스핀 파일도 현재 구조
위에서 열어 볼 수 있습니다.

CLI의 `analyze`에 대응하는 분석은 별도 버튼이 아니라 생성/실시간 갱신 때 자동으로
실행되며 오른쪽 `Analysis` 탭에 거리 shell, cutoff, 연결성, 이분성 및 layer 수로
표시됩니다.

Export에서는 DM.InitSpin 블록, 원본을 덮어쓰지 않는 패치된 SIESTA 입력, initial
magmom이 포함된 XYZ/CIF 구조를 저장할 수 있습니다. CLI가 기준 과학 구현이며 GUI
컨트롤러도 같은 코어 함수를 사용합니다.

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

## 테스트

```bash
python -m pytest
```

테스트는 FDF/include/patch, 1D·square·triangle·분리 graph, slab PBC, layer clustering, ordering, 1-based writer 및 validation 오류 검출을 다룹니다.
