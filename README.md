# InGaN Single Quantum Well / PL Simulation

nextnanopy로 nextnano++를 실행해 **Quantum well 밴드 그래프**와 **PL 발광 스펙트럼·peak**를 저장하는 Python 프로그램입니다. 실제 물리 계산은 nextnano++가 수행합니다.

## 준비

- Python 3.10 이상
- Python 라이브러리: `requirements.txt`
- 실제 계산: **nextnano++ 3.0.0 이상**, 해당 버전의 재료 database, 유효한 nextnano 라이선스

```bash
git clone https://github.com/Say1Won/PL_simulation.git
cd PL_simulation
python -m venv .venv
```

Windows에서는 `.venv\Scripts\activate`, Linux/macOS에서는 `source .venv/bin/activate`로 가상환경을 활성화합니다.

```bash
python -m pip install -r requirements.txt
```

nextnanomat에서 **Generate nextnanopy config file**로 실행 설정을 내보내거나, 다음과 같이 설정합니다. 아래 경로는 실제 설치 경로로 바꿉니다.

```python
import nextnanopy as nn
nn.config.set('nextnano++', 'exe', r'C:\path\nextnano++.exe')
nn.config.set('nextnano++', 'database', r'C:\path\database.nnp')
nn.config.set('nextnano++', 'license', r'C:\path\license.lic')
nn.config.save()
```

기본 설정 파일은 사용자 홈의 `.nextnanopy-config`입니다. 프로그램이 설정 파일을 자동 생성하거나 수정하지는 않습니다. 별도 설정 파일을 사용하려면 `--nextnano-config PATH`를 전달합니다. 실행 경로는 **절대 경로**를 사용하세요.

## 실행

```bash
python main.py
```

solver 없이 입력 변수·파일 준비를 확인하려면:

```bash
python main.py --prepare-only
```

이 명령은 설정을 검증하고 입력 파일을 저장합니다. nextnanopy의 변수 파싱 검사이며 **nextnano 물리 계산이나 solver 문법 검증을 대신하지 않습니다**.

이미 계산한 nextnano 결과를 분석하려면:

```bash
python main.py --results-dir "C:\path\nextnano_output"
```

`--results-dir`은 `bias_00000/`를 포함하는 solver 출력 폴더를 지정합니다. 사용한 구조·계산 조건에 맞는 `--structure`, `--settings`도 함께 지정하세요. 외부 결과의 실제 여기 조건을 현재 설정 파일만으로 검증할 수는 없습니다.

다른 설정/템플릿을 사용하거나 출력 위치를 바꿀 수 있습니다.

```bash
python main.py --structure configs/single_qw.json --settings configs/simulation.json --output-root ./outputs
python main.py --help
```

## 구조와 클래스

```text
PL_simulation/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── pl_simulation/
│   ├── __init__.py
│   ├── layer.py
│   ├── single_qw_structure.py
│   ├── simulation_settings.py
│   ├── nextnano_simulation.py
│   ├── qw_results.py
│   └── peak_analyzer.py
├── configs/
│   ├── single_qw.json
│   └── simulation.json
├── templates/
│   └── single_qw_pl.nnp
└── tests/
```

| 클래스 | 역할 |
|---|---|
| `Layer` | 재료·조성·두께·역할 검증 |
| `SingleQWStructure` | barrier–well–barrier 배치·경계 좌표·well 영역 |
| `SimulationSettings` | orientation·온도·mesh·캐리어 점유·스펙트럼 조건 |
| `NextnanoSimulation` | `InputFile`을 이용한 입력 수정·저장·실행 및 실패/수렴 확인 |
| `QWResults` | `DataFile`로 출력 읽기, 밴드 그래프 저장 |
| `PeakAnalyzer` | PL peak·FWHM 분석, 스펙트럼 그림·CSV·JSON 저장 |

클래스 파일은 각각 하나의 클래스만 정의합니다. import나 객체 생성만으로 solver를 실행하지 않습니다.

## 변경할 설정

`configs/single_qw.json`: 기본 구조는 **GaN 10 nm / In₀.₁₈Ga₀.₈₂N 3 nm / GaN 10 nm**입니다. barrier도 InGaN으로 설정할 수 있습니다. 현재 템플릿은 GaN substrate의 1D wurtzite 구조만 지원합니다.

`configs/simulation.json`의 `settings`:

- `temperature_k`, `grid_spacing_nm`, `electron_states`, `hole_states`
- `x_hkl`, `y_hkl`: nextnano의 wurtzite 축 설정. **성장면과 면내 기준면의 reduced 3-index Miller 지수**입니다. 네 개의 Miller–Bravais 지수를 그대로 넣지 않습니다. 기본 c-plane은 `[0,0,1]`, `[1,0,0]`입니다. 1D 계산 축은 simulation x입니다.
- `include_strain`, `include_polarization`: pseudomorphic strain, 압전·자발 분극 적용 여부
- `electron_fermi_ev`, `hole_fermi_ev`: 같은 solver 에너지 기준의 전자·정공 준페르미 준위
- `spectrum_energy_min_ev`, `spectrum_energy_max_ev`, `spectrum_energy_step_ev`, `broadening_ev`

orientation의 비영·비평행성은 Python이 검사하고, 실제 결정 격자 metric과 회전은 solver가 처리합니다. 방향에 따른 변화를 비교할 때 같은 substrate·여기 조건·수치 수렴 조건을 유지하세요.

`output_files`는 **출력 파일과 열을 명시적으로 지정**합니다. 기본값은 템플릿의 `single_qw`/`TEy` 이름을 사용합니다. solver 버전이나 템플릿을 변경해 파일 이름·열이 달라지면 이 매핑을 수정하세요. 여러 파일이 모호하게 일치하거나 필수 열이 없으면 실패하며 임의의 파일을 선택하지 않습니다.

## PL 모델과 결과 해석

기본 모델은 **지정한 준페르미 준위에서의 자발 방출 스펙트럼**입니다. 여기된 캐리어 점유를 입력으로 주고, strain·분극 및 Schrödinger–Poisson 전위/캐리어 screening과 unified **8-band k·p** 광학 계산을 연결합니다. 기본 3.0/0.0 eV 준페르미 값은 예시 조건입니다.

현재는 레이저 파장·파워에서 광생성률과 캐리어 포획·수명을 계산하는 모델을 포함하지 않습니다. 실측 레이저 파워에 대응하는 PL을 예측하려면 해당 과정이나 별도로 검증한 캐리어 점유 조건을 연결해야 합니다. 기본 스펙트럼은 simulation y 방향 `TEy` 편광이며 전체 편광·각도 적분 발광량은 아닙니다. 기본 템플릿의 k-space 적분·굴절률 설정도 수렴/보정 대상입니다.

밴드 그래프의 valence 곡선은 `HH`, `LH`, `SO` 중 위치별 최대값입니다. 고유준위/파동함수는 명시적 출력 매핑이 있을 때 `--include-states`/`--include-wavefunctions`로 추가합니다. kp8 상태를 임의로 전자/정공으로 분류하지 않습니다. 상태를 지정할 때 conduction-like/valence-like 성분을 확인하고 **동일한 에너지 기준의 valence electron energy**를 사용하세요. 파동함수는 표시용 크기로 스케일합니다.

기본 PL 출력은 solver의 photon density per eV를 읽고 `|dE/dλ| = hc/λ²`를 적용해 **per nm**로 변환합니다. 따라서 파장 스펙트럼 peak는 에너지 스펙트럼 peak의 단순 좌표 변환과 다를 수 있습니다. `estimate_transition()`의 준위 간 전이 추정도 PL peak와 별도입니다.

FWHM은 선택된 peak 주변의 연속적인 반치폭을 선형 보간해 구합니다. baseline을 빼지 않으며, peak/반치 구간이 출력 경계에서 잘리면 `null`로 기록합니다. 전부 0인 스펙트럼에서는 peak를 만들어 내지 않고 여기 조건·파일 매핑 확인을 요청합니다.

## 생성 결과

`outputs/`는 Git에 포함하지 않으며 처음 실행할 때 자동 생성됩니다. 기존 결과를 덮어쓰지 않고 `run_001`, `run_002`, …로 저장합니다.

```text
outputs/run_001/
├── inputs/       # 적용된 .nnp, 구조·설정 사본, 실행 상태
├── nextnano/     # 실제 solver 출력 및 로그
├── figures/
│   ├── quantum_well.png
│   └── pl_spectrum.png
└── analysis/
    ├── pl_spectrum.csv
    └── pl_summary.json
```

`--prepare-only`는 `inputs/`까지만 생성합니다. `--results-dir`은 원본 데이터를 복사하지 않고 분석 파일만 새 run에 저장합니다. 실패한 실행은 `inputs/run.json`에 원인을 기록합니다.

## 검증 및 MQW 확장

```bash
python -m unittest discover -s tests -v
```

검증은 구조/단위/입력 준비와 시험용 데이터의 파일 읽기·분석·그림 저장을 포함합니다. **이 저장소의 초기 구현은 라이선스가 있는 nextnano++에서 전체 물리 계산을 실행해 검증하지 않았습니다.** 실제 사용 시 solver 로그, 상태 분류 및 mesh·상태 수·k 적분·에너지 범위의 수렴을 확인하세요. 시험용 데이터는 물리 시뮬레이션 결과가 아닙니다.

MQW 확장 시 `MQWStructure`, MQW 템플릿과 설정을 추가하고 기존 실행·결과·분석 클래스를 재사용합니다. 전체 MQW를 함께 풀어 well 사이 결합을 반영해야 합니다.

참고: [nextnanopy 설정](https://www.nextnano.com/docu/nextnanopy/getting_started/configuration.html), [결정 좌표계](https://www.nextnano.com/docu/nextnanoplus/latest/reference/models/strain/crystal_coordinate_systems.html), [quantum optical spectra](https://www.nextnano.com/docu/nextnanoplus/latest/reference/keywords/optics/quantum_spectra.html), [QW PL tutorial](https://www.nextnano.com/docu/nextnanoplus/latest/tutorials/quantum_well_photoluminescence_resonant.html)
