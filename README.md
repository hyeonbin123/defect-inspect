# defect-inspect

정상 제품 사진만으로 만드는 외관 검사기(이상 탐지)를, 현장에서 실제로 굴릴 때 부딪히는 질문으로 재는 프로젝트다.

공개 벤치마크의 AUROC는 이미 포화에 가깝다. 그래서 점수 경쟁 대신 다음을 미리 정한 규칙으로 잰다.

1. 결함 라벨 몇 장부터 지도 학습이 비지도 이상 탐지를 이기는가
2. 정상 이미지로 고정한 임계값의 실제 오검출률은 목표와 얼마나 다른가. 그 차이는 표본이 유한해서 생기는 흔들림인가, 절차의 치우침인가
3. 밝기·흐림 같은 합성 교란은 실제 조명 변화를 대신할 수 있는가
4. CPU에서 장당 200ms 안에 들어오면서 오검출률 기준을 지키는 구성이 남는가

계획은 [docs/plan.md](docs/plan.md), 측정 규칙과 결과는 [docs/experiments.md](docs/experiments.md)에 있다. 규칙은 재기 전에 커밋한다.

## 지금까지의 결과
1단계까지 쟀다. 자세한 표와 판정은 [docs/experiments.md](docs/experiments.md)에 있다.

**기준선 (PatchCore, WideResNet-50, 256px).** VisA 공식 `2cls_highshot` 테스트(정상 3,848장, 결함 480장)에서 평균 이미지 AUROC 89.3 [87.8, 90.8], AUPRO 90.5. 결함의 95%를 잡으려면 정상의 34%를 불량으로 돌려야 한다.

**임계값을 정상 이미지만으로 정하는 세 가지 방법** (목표 오검출률 5%, 12개 범주 합산)

| 방식 | 실제 오검출률 | 검출률 | 미리 적어 둔 가설 | 판정 |
|---|---|---|---|---|
| 재대입: 뱅크를 만든 이미지의 점수로 임계값을 정함 | 86.0% | 100% | 목표의 두 배 이상이다 | 지지 |
| 홀드아웃: 빼 둔 정상 20%의 점수로 정함 | 5.6% | 68.1% | 이론 구간(3.6~6.3%) 안이다 | 지지 |
| 교차 적합: 겹마다 "그 겹을 뺀 뱅크"로 매긴 점수를 모아 정함 | 4.0% | 66.7% | 목표를 넘지 않는다 | 지지 |

- 뱅크를 만든 이미지로 임계값을 잡으면 정상의 86%가 불량으로 나온다
- 빼 둔 정상으로 잡으면 표본이 범주당 60~121장뿐이어도 이론이 말하는 범위에 들어온다
- 교차 적합은 1%p쯤 보수적이다. 같은 목표에서 어느 쪽이 결함을 더 잡는지는 가리지 못했다(차이 −1.5%p [−2.9, 0.0], 판정 불가)
- 오검출률을 5%로 묶으면 결함의 약 3분의 2를, 1%로 묶으면 절반에 못 미치게 잡는다

2단계(백본 교체, Dinomaly, 결함 라벨 k장 지도 학습과의 비교)부터는 진행 중이다.

## 데이터와 라이선스
| 이름 | 쓰임 | 라이선스 | 근거 |
|---|---|---|---|
| [VisA](https://github.com/amazon-science/spot-diff) | 주 데이터 | CC BY 4.0 | 공식 저장소 README의 License 절, [LICENSE-DATASET](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET), [AWS Open Data Registry](https://registry.opendata.aws/visa/) |
| [M2AD](https://huggingface.co/datasets/ChengYuQi99/M2AD) | 실제 조명 변화 (3단계, 2개 범주) | Apache-2.0 | Hugging Face 데이터셋 카드의 license 표기 |

데이터는 저장소에 넣지 않는다. 다운로드 스크립트와 분할 매니페스트(파일 경로 목록)만 둔다.

VisA: Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation", ECCV 2022.

## 설치와 테스트
```bash
uv sync                                   # 지표·분할·임계값 보정 코드 (numpy, scipy, pillow)
uv sync --group serve                     # 검사 서비스까지 (onnxruntime, FastAPI; torch 없음)
uv sync --group train --group dinomaly --group serve   # 특징 추출·학습·내보내기까지 (torch CUDA 12.8 빌드)
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## 재현 명령
```bash
# 0단계: 데이터와 분할
uv run python -m defect_inspect.download               # VisA tar (1.93GB, sha256 검증)
uv run python -m defect_inspect.splits                 # manifests/visa.csv
uv run python -m defect_inspect.cache --size 256       # 리사이즈 캐시 (448, 392 등도 같은 방식)

# 1단계: 기준선과 임계값 보정 (dev로 점검한 뒤 test는 한 번)
uv run python -m defect_inspect.run_patchcore --config p0 --protocol dev
uv run python -m defect_inspect.analyze outputs/p0-dev --stage stage1
uv run python -m defect_inspect.run_patchcore --config p0 --protocol test --allow-test --stage 1 --save-bank

# 2단계: 다른 백본, Dinomaly, 지도 학습
uv run python -m defect_inspect.run_patchcore --config d-s --protocol test --allow-test --stage 2 --save-bank
uv run python -m defect_inspect.run_dinomaly train --no-amp --batch-size 8
uv run python -m defect_inspect.run_dinomaly eval --protocol test --allow-test --stage 2
uv run python -m defect_inspect.run_supervised --protocol test --allow-test --stage 2
uv run python -m defect_inspect.compare --protocol test --dinomaly --supervised --reference dm

# 3단계: 합성 교란(VisA)과 실제 조명 변화(M2AD)
uv run python -m defect_inspect.run_perturb --method p0 --protocol test --allow-test --stage 3
uv run python -m defect_inspect.analyze_perturb --protocol test
uv run python -m defect_inspect.m2ad --size 256        # data/raw/m2ad 의 zip에서 캐시 생성
uv run python -m defect_inspect.run_m2ad --method p0 --allow-test --stage 3-m2ad
uv run python -m defect_inspect.analyze_m2ad --methods p0 d-s

# 4단계: 코어셋·해상도 격자, 내보내기, 지연
uv run python -m defect_inspect.run_grid --backbone wrn50 --size 256 --protocol dev --save-banks
uv run python -m defect_inspect.export --grid outputs/grid-wrn50-256-dev --ratio 0.01 --out artifacts/wrn50-256-r0.01
uv run python -m defect_inspect.bench --artifacts artifacts/wrn50-256-r0.01 --key wrn50-256-r0.01 --gpu
uv run python -m defect_inspect.analyze_grid --protocol dev
```
`--protocol test`나 `--allow-test`가 붙은 실행은 봉인 테스트를 읽고, 읽을 때마다 `reports/test_ledger.jsonl`에 한 줄을 남긴다.

## 검사 서비스
torch 없이 onnxruntime만으로 CPU에서 돈다. `defect_inspect.export`가 만든 아티팩트 폴더(ONNX 모델, 범주별 메모리 뱅크와 임계값)를 읽는다.

```bash
# 모델 파일 없이 도는 합성 범주(demo)로 띄워 보기
uv run uvicorn defect_inspect.service:create_offline_app --factory --port 8093

# 아티팩트로 띄우기
DEFECT_INSPECT_ARTIFACTS=artifacts/serving uv run uvicorn defect_inspect.service:create_app --factory --port 8093

# Docker (CPU 이미지, 약 650MB)
DEFECT_INSPECT_ARTIFACT_DIR=./artifacts/serving docker compose up --build
```

| 경로 | 내용 |
|---|---|
| `GET /` | 데모 페이지: 사진을 올리면 판정과 이상 위치를 겹쳐 보여 준다 |
| `POST /inspect` | `image`(파일), `category` → 이상 점수, 임계값, 양품/불량, 히트맵 PNG(base64) |
| `POST /calibrate` | 현재 촬영 조건의 정상 사진 여러 장 → 그 범주의 임계값을 다시 잡는다. `DEFECT_INSPECT_ADMIN_TOKEN`을 설정하고 `X-Admin-Token` 헤더로 보내야 하며, 설정하지 않으면 꺼져 있다 |
| `DELETE /calibrate` | 재보정을 되돌린다 |
| `GET /categories`, `GET /healthz` | 범주별 임계값·뱅크 크기, 상태 |

- 받는 형식은 PNG, JPEG, BMP, TIFF, WebP(채널당 8비트, 투명도 없음)이고 업로드는 파일당 20MB까지다. 16비트 이미지는 조용히 잘리지 않게 거절한다
- 재보정한 임계값은 메모리에만 있다. 다시 띄우면 아티팩트의 값으로 돌아간다
- 기본으로 `localhost`와 `127.0.0.1` 이름으로만 응답한다. 다른 이름이나 주소로 열려면 `DEFECT_INSPECT_ALLOWED_HOSTS`에 적는다
