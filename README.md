# InGaN Single Quantum Well / PL Simulation

Python으로 **Quantum well 밴드·고유준위 그래프**와 **상대 PL 스펙트럼·peak**를 계산합니다. nextnano++ 설치, database, 라이선스가 필요하지 않습니다. 결과는 사용자가 직접 측정한 실험 데이터와 비교할 수 있도록 PNG·CSV·JSON으로 저장합니다.

현재 계산은 **c-plane, unstrained 1D 단일 밴드 유효질량 모델**입니다. 전자와 정공의 구속 상태를 직접 풀고, 실제 envelope overlap과 캐리어 점유에서 발광 스펙트럼을 계산합니다. 실험값과 자동으로 일치시키거나 절대 PL 세기를 예측하는 모델은 아닙니다.

## 설치 및 실행

Python 3.10 이상과 Git을 준비합니다.

```bash
git clone https://github.com/Say1Won/PL_simulation.git
cd PL_simulation
python -m venv .venv
```

Windows 명령 프롬프트(cmd):

```bat
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

라이브러리를 설치하고 실행합니다.

```bash
python -m pip install -r requirements.txt
python main.py
```

기본 실행에는 NumPy·SciPy·Matplotlib만 사용합니다. `nextnanopy` import, 외부 solver 호출, nextnano 기준 결과와의 대조를 수행하지 않습니다.

계산 없이 입력 설정만 확인하려면:

```bash
python main.py --prepare-only
```

파동함수를 밴드 그래프에 함께 표시하려면:

```bash
python main.py --include-wavefunctions
```

계산한 아카이브를 다시 분석하려면:

```bash
python main.py --results-dir outputs/run_001/calculation
```

`.npz` 파일을 직접 지정해도 됩니다. 아카이브의 원래 구조·계산 조건·재료 상수를 결과 metadata에서 확인할 수 있습니다. 다시 분석할 때 solver 계산은 수행하지 않습니다.

다른 설정·출력 폴더는 다음과 같이 지정합니다.

```bash
python main.py --structure configs/single_qw.json --settings configs/simulation.json --materials configs/materials.json --output-root ./outputs
python main.py --help
```

## 파일 구성과 클래스

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
│   ├── quantum_well_simulation.py
│   ├── qw_results.py
│   ├── peak_analyzer.py
│   └── material_parameters.py
├── configs/
│   ├── single_qw.json
│   ├── simulation.json
│   └── materials.json
└── tests/
```

| 클래스 | 역할 |
|---|---|
| `Layer` | 재료·조성·두께·역할 검증 |
| `SingleQWStructure` | barrier–well–barrier 배치와 공간 경계 |
| `SimulationSettings` | 온도·mesh·점유 밀도·전기장·스펙트럼 조건 |
| `QuantumWellSimulation` | Python 고유상태·준페르미 준위·상대 PL 계산 |
| `QWResults` | 계산 배열·아카이브 읽기, 밴드·고유준위 그래프 |
| `PeakAnalyzer` | PL peak·FWHM 분석, 그림·CSV·JSON 저장 |

클래스 파일은 각각 클래스 하나를 정의합니다. `material_parameters.py`는 재료 상수를 읽고 밴드와 질량 배열을 만드는 함수 모듈입니다. 기존 nextnano 실행 클래스·입력 템플릿은 제거했습니다.

## 변경할 설정

`configs/single_qw.json`의 기본 구조는 **GaN 10 nm / In₀.₁₈Ga₀.₈₂N 3 nm / GaN 10 nm**입니다. barrier는 well보다 낮은 In 조성의 InGaN으로도 지정할 수 있습니다.

`configs/simulation.json`의 `settings`:

- `temperature_k`, `grid_spacing_nm`: 온도(K)와 요청 mesh 간격(nm). 전체 구조 길이에 맞춘 실제 균일 간격은 metadata에 기록합니다.
- `electron_states`, `hole_states`: 구할 저에너지 상태 수. 장벽 아래의 구속 상태만 캐리어 점유와 PL에 사용합니다.
- `electron_sheet_density_cm2`, `hole_sheet_density_cm2`: well 구속 상태에 할당한 면밀도(cm⁻²). 기본값은 각각 `1e12`이며, 준페르미 준위는 내부에서 계산합니다.
- `conduction_band_offset_ratio`: gap 차이 중 conduction offset에 배분할 비율. 기본 `0.7`은 변경 가능한 모델 가정입니다.
- `electric_field_kv_cm`: 성장 좌표를 따라 사용자가 지정한 균일 전기장(kV/cm). 기본 `0`. 양의 값은 conduction·valence 전자 에너지를 성장 좌표에 따라 올립니다.
- `spectrum_energy_min_ev`, `spectrum_energy_max_ev`, `spectrum_energy_step_ev`: 에너지 스펙트럼 범위와 샘플 간격.
- `broadening_ev`: **Gaussian 표준편차 σ(eV)**. 선폭은 약 `2.355σ`이며, 발광 스펙트럼 전체 FWHM은 캐리어 점유와 여러 전이에도 영향을 받습니다.
- `k_integration_points`: 면내 k 적분의 최소 점 수. 좁은 broadening을 해상할 수 있도록 필요한 경우 내부에서 점 수를 늘립니다.
- `x_hkl`, `y_hkl`: 현재는 c-plane 성장 `±[0,0,1]`과 비영 basal-plane 기준만 지원합니다. 다른 orientation은 오류로 거부합니다.
- `include_strain`, `include_polarization`: 현재 반드시 `false`. 지원하지 않는 물리를 활성화하면 명시적으로 실패합니다.

예전 `electron_fermi_ev`·`hole_fermi_ev` 설정은 사용하지 않습니다. 새 기본 설정 파일을 사용하세요. 기존 값에서 면밀도를 자동 추정하지 않습니다.

## 재료 상수와 계산 모델

`configs/materials.json`에서 GaN/InN gap·Varshni 계수·유효질량·InGaN gap bowing과 논문 출처를 변경할 수 있습니다. 기본 상수는 Vurgaftman & Meyer, *Journal of Applied Physics* **94**, 3675 (2003), [DOI:10.1063/1.1600519](https://doi.org/10.1063/1.1600519)의 구분 가능한 2003 parameter set입니다. InN gap과 bowing은 문헌·시료에 따라 달라질 수 있으므로 다른 값을 사용하면 출처도 함께 수정하세요.

조성에 따른 gap은 온도 의존 GaN/InN gap의 보간과 bowing으로 구합니다. 유효질량은 선형 보간합니다. 정공 질량은 wurtzite valence Hamiltonian의 **scalar A-like diagonal projection**이며, 전체 다중 밴드 계산을 대체하지 않습니다. 유도식과 원래 A 계수는 JSON에 기록했습니다.

계산 순서:

1. 위치별 conduction·valence 밴드와 성장 방향·면내 유효질량 생성.
2. 위치 의존 질량을 반영한 BenDaniel–Duke Hamiltonian의 저에너지 고유상태 계산. 바깥 경계는 `ψ=0`입니다.
3. 양쪽 장벽보다 낮은 상태를 선택하고, 2D parabolic subband 점유가 지정 면밀도를 재현하도록 전자·정공 준페르미 준위 계산.
4. 전자·정공 envelope overlap, 면내 joint density of states, Fermi 점유를 적분하여 상대 발광 스펙트럼 계산. 일정한 interband momentum matrix element를 가정하고 photon energy factor를 사용합니다.
5. 면적이 1인 Gaussian kernel로 broadening, 에너지 밀도를 파장 밀도로 변환, peak·FWHM 분석.

tridiagonal 고유해석으로 필요한 상태만 계산하고, 면내 적분은 묶어서 처리해 큰 임시 배열을 피합니다. 이미 읽은 밴드·스펙트럼은 그래프 생성에 재사용합니다.

현재 자동 strain·자발/압전 분극, self-consistent Poisson·carrier screening, valence mixing, exciton, alloy localization, 포획·수명·비방사 재결합 및 검출기 응답은 포함하지 않습니다. 외부 전기장 입력은 이러한 물리의 자동 계산과 별개입니다. c-plane InGaN 실험에서 이 효과들이 peak·선폭·강도 차이의 원인이 될 수 있습니다.

## 생성 결과와 실험 비교

`outputs/`는 Git에서 제외하며 실행 시 자동 생성합니다. 기존 결과를 덮어쓰지 않고 `run_001`, `run_002`, …로 저장합니다.

```text
outputs/run_001/
├── inputs/
│   ├── single_qw.json
│   ├── simulation.json
│   ├── materials.json
│   └── run.json
├── calculation/
│   └── result.npz
├── figures/
│   ├── quantum_well.png
│   └── pl_spectrum.png
└── analysis/
    ├── pl_spectrum.csv
    └── pl_summary.json
```

`quantum_well.png`에는 밴드와 계산한 전자·valence 전자 고유준위를 함께 표시합니다. 정공 quasiparticle 고유에너지를 valence 전자 에너지로 변환해 같은 기준에서 표시합니다. `--include-wavefunctions`의 세로 크기는 표시용 스케일입니다.

`pl_spectrum.csv`는 파장(nm)과 **상대 스펙트럼 밀도 per nm**입니다. 에너지 밀도 per eV에서 `|dE/dλ|=hc/λ²`를 적용하므로 파장 peak는 에너지 peak의 단순 좌표 변환과 다를 수 있습니다. 절대 photons/s·레이저 파워 환산 값은 아닙니다.

실험과의 peak·선폭·스펙트럼 모양 비교에는 CSV와 `pl_summary.json`을 사용하세요. 실험의 파장축·에너지축과 밀도 단위를 맞추고, 세기를 정규화했다면 동일한 정규화 방식을 적용하세요. 비교용 외부 solver 실행이나 실험 데이터 자동 fitting은 포함하지 않습니다.

FWHM은 선택된 peak 주변의 연속적인 반치 구간을 선형 보간하며 baseline을 빼지 않습니다. 반치 구간이 계산 범위에서 잘리면 `null`, 전부 0인 데이터는 실패로 처리합니다. 아카이브에는 고유상태·원래 에너지 스펙트럼과 모델·밀도 잔차·재료 출처가 저장됩니다. `.npz`는 pickle을 사용하지 않습니다.

`--prepare-only`는 `inputs/`만 생성합니다. 실패 원인은 해당 실행의 `inputs/run.json`에 기록합니다.

## 선택적 기존 파일 가져오기

기존 nextnano `.dat` 파일을 읽고 싶을 때만 `nextnanopy.DataFile`을 사용할 수 있습니다.

```bash
python -m pip install nextnanopy==1.4.0
```

`configs/simulation.json`에 `output_files`의 `band_edges`·`spectrum` 파일/열/단위 매핑을 명시하고 `--results-dir`에 기존 데이터 폴더를 지정합니다. 기본 Python 계산에는 이 설정이나 라이브러리가 필요하지 않습니다. 여러 파일에 모호하게 일치하는 매핑은 거부합니다.

## 수치 검증 및 MQW 확장

```bash
python -m unittest discover -s tests -v
```

무한 우물 해석해, 질량 경계와 에너지 기준 이동, 파동함수 정규직교성, 지정 밀도 재현, 스펙트럼 Jacobian의 적분 보존, 아카이브 저장·복원 및 전체 Python 실행을 검사합니다. nextnanopy가 없으면 선택적 기존 `.dat` 테스트만 건너뜁니다. 다른 solver 결과와 비교하는 테스트는 없습니다.

실제 시료와 비교하기 전에 mesh·barrier 길이·상태 수·k 적분·스펙트럼 샘플링을 변화시켜 수렴을 확인하세요. 얕은 장벽 근처의 상태 수는 mesh에 민감할 수 있습니다. 이런 수치 확인은 실험과의 물리 모델 검증과 별개입니다.

MQW는 향후 `MQWStructure`를 추가하고 전체 영역을 함께 풀어 well 사이 결합을 반영하도록 확장할 수 있습니다. 현재 `SingleQWStructure`는 정확히 하나의 well만 허용합니다.
