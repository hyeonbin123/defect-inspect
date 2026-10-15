"""CI의 Docker 스모크 검사용 작은 복원 방식 아티팩트 세트를 만든다.

`python .github/ci/toy_artifact.py <폴더> [--wrong-digest]`
"""

# [파일 역할] GitHub Actions의 docker 작업이 서비스 이미지를 실제 기본 명령(`create_app`, `/artifacts`를
# 읽음)으로 띄워 보려고 쓰는 도우미다. 학습한 모델(약 134MB)은 저장소에 없으므로, 서빙 모델과 같은
# 입출력(입력 `image` [1, 3, S, S], 출력 `score` [1]과 `map` [1, M, M])을 가진 아주 작은 ONNX 그래프(맵 =
# 채널 평균, 점수 = 맵의 최댓값)와 범주 폴더 하나(`toy/meta.json`)를 만든다. 모양은 `dinomaly_serving
# export`가 만드는 세트와 같다: 세트 루트에 모델 하나, 범주 폴더에는 meta.json만.
# [왜 필요한가] 합성 범주로만 도는 오프라인 앱(`create_offline_app`)은 ONNX 모델을 열지 않아, 이미지 안의
# onnxruntime·아티팩트 읽기·모델 지문 대조(`inspector.check_model_sha256`) 경로를 거치지 않는다. 이 세트로
# 띄우면 컨테이너 안에서 그 경로가 실제로 돈다. `--wrong-digest`는 meta에 다른 지문을 적어, 서비스가 시작을
# 거절하는지 보는 데 쓴다.
# [입력·출력] 출력 폴더에 model_fp32.onnx, toy/meta.json을 쓴다. 표준 출력에는 모델 SHA-256을 찍는다.
# [주의] onnx 패키지가 필요하다(서비스 이미지·CI 테스트 환경에는 없으므로 docker 작업이 uv로 잠시 더한다).
# 그래프는 tests/test_service_reconstruction.py의 `_toy_onnx`와 같은 것이다.

import argparse
import hashlib
import json
from pathlib import Path

# SIZE: 모델 입력 크기(px). 작은 그래프라 업로드를 이 크기로 줄여도 빠르다. 맵도 같은 크기다.
SIZE = 32


# 서빙 인터페이스를 가진 ONNX 그래프를 저장한다. onnx.checker로 형식을 확인한다(onnxruntime이 읽을 수 있게).
def write_model(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, SIZE, SIZE])
    score = helper.make_tensor_value_info("score", TensorProto.FLOAT, [1])
    amap = helper.make_tensor_value_info("map", TensorProto.FLOAT, [1, SIZE, SIZE])
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["map"], axes=[1], keepdims=0),
        helper.make_node("ReduceMax", ["map"], ["score"], axes=[1, 2], keepdims=0),
    ]
    graph = helper.make_graph(nodes, "toy", [image], [score, amap])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


# 세트를 만들고 모델 지문을 돌려준다. wrong_digest면 meta에 일부러 다른 지문(0 64개)을 적는다.
# 임계값 0.5: ImageNet 정규화 뒤 채널 평균의 최댓값이라, 밝은 칸이 있으면 넘고 회색 사진은 넘지 않는다.
def build(out_dir: Path, wrong_digest: bool = False) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = out_dir / "model_fp32.onnx"
    write_model(model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    meta = {
        "version": 1,
        "kind": "reconstruction",
        "name": "ci-toy",
        "category": "toy",
        "img_size": SIZE,
        "map_size": SIZE,
        "threshold": 0.5,
        "alpha": 0.05,
        "calibration": {"strategy": "none"},
        "source": {"onnx_sha256": "0" * 64 if wrong_digest else digest},
    }
    (out_dir / "toy").mkdir(exist_ok=True)
    with open(out_dir / "toy" / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CI Docker 스모크 검사용 작은 복원 방식 아티팩트 세트를 만든다."
    )
    parser.add_argument("out", type=Path, help="세트를 쓸 폴더(없으면 만든다)")
    parser.add_argument(
        "--wrong-digest",
        action="store_true",
        help="meta.json에 실제 모델과 다른 지문을 적는다(서비스가 시작을 거절하는지 볼 때)",
    )
    args = parser.parse_args()
    print(build(args.out, args.wrong_digest))


if __name__ == "__main__":
    main()
