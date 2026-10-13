"""Changes to anomalib's DinomalyModel for the stage 6 models, and the graph that is exported for the CPU.

Needs torch and anomalib (the `train` and `dinomaly` groups). Every change keeps the model's outputs:

- `fix_encoder` keeps only the encoder blocks the model reads (the last target layer is block 9; anomalib
  runs all 12 blocks of the ViT) and resamples the position embedding once for a fixed input size, so
  that no interpolation runs per image. The same resampling function runs as in the dynamic path.
- `ScaledLinearAttention` divides the keys by the number of tokens. The output of linear attention is
  `(q @ (k^T v)) / (q . sum(k))`: the factor cancels, but the sums that overflowed fp16 in stage 2
  (`work/dinomaly_nan_probe_cpu.json`: q.kv up to 89,101) shrink by that factor.
- `Fp32LinearAttention` runs the attention in fp32 with autocast switched off (the fallback of the
  fp16 gate in docs/experiments.md, stage 6).
- `ServingGraph` is the inference path of `DinomalyModel.forward` plus the map resize of
  `run_dinomaly.predict`: image in, (image score, 256x256 map) out.
"""

# [파일 역할] 6단계 CPU 서빙 모델(Dinomaly ViT-S, DMS)을 위해 anomalib의 DinomalyModel을 고치는 도구와, CPU용
# ONNX로 내보낼 추론 그래프(ServingGraph)를 모은 모듈이다.
# [왜 필요한가] 4단계 PatchCore 서빙 구성은 CPU 200ms와 정확도 기준을 함께 지키지 못했다(H15 기각). 6단계는
# Dinomaly의 인코더를 ViT-S로 줄여 CPU에서 돌리려 했고, 그러려면 (1) 쓰지 않는 계산을 덜어 내고, (2) fp16
# 학습이 깨지지 않게 하고, (3) 점수 계산까지 한 그래프로 내보내야 했다.
# [주요 구성] docstring의 네 가지: fix_encoder(인코더 블록 0~9만 남기고 입력 크기 고정),
# ScaledLinearAttention(키를 토큰 수로 나눠 fp16 넘침 방지), Fp32LinearAttention(관문 2의 대안),
# ServingGraph(이미지 → 점수와 256×256 맵). 모두 출력을 바꾸지 않는 변경이고, 테스트와 관문 2 (가)로 원본과
# 같은 출력을 확인했다(docs/experiments.md 6단계: 점수 상대 차이 0, 맵 2.4e-7).
# [연결] run_dinomaly.build_model이 설정에 따라 fix_encoder와 set_attention을 부르고, dinomaly_serving의 ONNX
# 내보내기가 ServingGraph를 쓴다. torch와 anomalib(train·dinomaly 의존성 그룹)이 있어야 해서 서빙 쪽은 이
# 모듈을 import하지 않는다.

from __future__ import annotations

# torch는 모델 수정과 그래프 계산에 쓰고, anomalib 2.6.2에서는 Dinomaly 디코더의 선형 어텐션 클래스와 점수
# 계산 상수(맵 크기 DEFAULT_RESIZE_SIZE, 상위 비율 DEFAULT_MAX_RATIO)를 가져온다. 모델 코드는 anomalib 것을
# 쓰고 학습 루프만 직접 쓰는 것이 이 프로젝트의 방식이다(run_dinomaly.py docstring).
import torch
import torch.nn.functional as F
from anomalib.models.image.dinomaly.components import LinearAttention
from anomalib.models.image.dinomaly.torch_model import DEFAULT_MAX_RATIO, DEFAULT_RESIZE_SIZE

# 디코더 어텐션의 세 가지 모드: original(anomalib 그대로), scaled(키를 N으로 나눈 패치), fp32(어텐션만 fp32).
# 6단계 관문 2 (나)에서 scaled + fp16이 통과해 모든 후보를 이 조합으로 학습했다. MAP_SIZE는 픽셀 지표와 서빙
# 히트맵의 격자(256×256)다.
ATTENTION_MODES = ("original", "scaled", "fp32")
MAP_SIZE = 256


# fp16에서 안전한 선형 어텐션. 2단계 DM(ViT-B 392px)은 fp16 학습 15스텝째에 손실이 NaN이 되었는데, 원인으로 본
# 것이 디코더 선형 어텐션의 합이 fp16 최댓값 65,504를 넘는 것이었다(fp32 점검에서 q·kv 최대 89,101).
# 선형 어텐션의 출력은 (q·(kᵀv)) / (q·Σk)라 키 k를 토큰 수 N으로 나눠도 분자·분모에서 N이 약분되어 수식상
# 같고, 중간 합만 1/N로 작아진다. 두 번째 반환값 kv는 원래의 1/N이지만 디코더 블록은 쓰지 않는다.
# 주의(2026-10-04 정정): 이 패치로 ViT-S 280px가 fp16에서 유한하게 학습된 것은 보였지만, 패치가 2단계 DM의
# fp16 실패를 없앤다는 것은 재지 않았다.
class ScaledLinearAttention(LinearAttention):
    """anomalib's LinearAttention with the keys divided by the number of tokens.

    The output is the same up to rounding; the key-value product it returns as its second value (which
    the decoder block does not use) is the original one divided by the number of tokens.
    """

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # anomalib 코드와 달라진 곳은 `k = (F.elu(k) + 1.0) / seq_len` 한 줄뿐이다. elu(x) + 1은 선형
        # 어텐션에서 softmax 대신 쓰는 양수 특징 사상이라, 분모 q·Σk가 0이 되지 않는다.
        # The body of anomalib 2.6.2's LinearAttention.forward (Apache-2.0) with `k / seq_len`.
        batch_size, seq_len, embed_dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = (F.elu(k) + 1.0) / seq_len
        kv = torch.matmul(k.transpose(-2, -1), v)
        k_sum = k.sum(dim=-2, keepdim=True)
        z = 1.0 / torch.sum(q * k_sum, dim=-1, keepdim=True)
        x = torch.matmul(q, kv) * z
        x = x.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


# 관문 2 (나)의 대안: 다른 층은 fp16 autocast로 돌리되 선형 어텐션만 autocast를 끄고 fp32로 계산한다. 첫
# 변형(scaled)이 통과해 실제 학습에는 쓰이지 않았다(docs/experiments.md 6단계 관문 2 (나) 결과).
class Fp32LinearAttention(LinearAttention):
    """anomalib's LinearAttention computed in fp32, outside any autocast region."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float())


# 모드 이름 → 어텐션 클래스. 세 클래스는 파라미터 구성이 같고 forward만 다르다.
_ATTENTION = {"original": LinearAttention, "scaled": ScaledLinearAttention, "fp32": Fp32LinearAttention}


# 모든 디코더 블록의 어텐션 객체의 클래스(__class__)만 바꿔 끼운다. 새 모듈을 만들지 않으므로 파라미터 객체와
# state_dict 키가 그대로라, 원본 어텐션으로 학습한 가중치와 패치한 모델이 서로 호환된다(관문 2 (가)는 같은
# 초기값에서 두 모델을 비교했다). 디코더에 LinearAttention이 아닌 것이 있으면 TypeError를 낸다. 호출 위치:
# run_dinomaly.build_model.
def set_attention(model: torch.nn.Module, mode: str) -> None:
    """Switch the attention of every decoder block to `mode` (the parameters stay the same objects)."""
    if mode not in _ATTENTION:
        raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {mode!r}")
    for block in model.decoder:
        if not isinstance(block.attn, LinearAttention):
            raise TypeError(f"decoder attention is {type(block.attn).__name__}, not a LinearAttention")
        block.attn.__class__ = _ATTENTION[mode]


# 지금 디코더가 어느 어텐션 모드인지 돌려준다. 블록마다 다르면 ValueError.
# 테스트(tests/test_dinomaly_model.py, test_run_dinomaly.py)가 저장·불러오기 뒤에도 모드가 유지되는지 확인하는
# 데 쓴다.
def attention_mode(model: torch.nn.Module) -> str:
    """The mode of the decoder's attention (all blocks must agree)."""
    modes = {next(k for k, cls in _ATTENTION.items() if type(b.attn) is cls) for b in model.decoder}
    if len(modes) != 1:
        raise ValueError(f"decoder blocks use different attention modes: {sorted(modes)}")
    return modes.pop()


# 입력 크기를 고정한 인코더에 붙일 forward 사전 훅을 만든다. 다른 크기 이미지가 들어오면 이미 보간한 위치
# 임베딩을 다시 보간하지 않고 오류로 거절한다(docs/experiments.md 6단계: "고정한 모델에 다른 크기를 넣으면
# 오류가 난다").
def _size_guard(img_size: int):
    def check(module: torch.nn.Module, args: tuple) -> None:
        size = tuple(args[0].shape[-2:])
        if size != (img_size, img_size):
            raise ValueError(
                f"this encoder is fixed to {img_size}x{img_size} inputs, got {size[0]}x{size[1]}"
            )

    return check


# 6단계 인코더 경량화. 둘 다 출력을 바꾸지 않는다.
# - blocks: 앞쪽 블록만 남긴다. Dinomaly가 읽는 층은 2~9라 10, 11번 블록은 계산만 하고 출력이 쓰이지 않는다.
#   그래서 10블록(DMS 설정의 encoder_blocks=10)으로 자른다. 읽는 마지막 층보다 적게 자르면 ValueError.
# - img_size: 위치 임베딩을 입력 크기에 맞게 한 번만 보간해 둔다. 동적 경로가 이미지마다 부르는 같은
#   함수(anomalib가 antialias를 끈 bicubic)로 계산하므로 결과가 비트 단위로 같고, 이미지마다의 보간 연산이
#   ONNX 그래프와 CPU 추론에서 빠진다. 크기는 패치(14)의 배수여야 한다.
def fix_encoder(model: torch.nn.Module, *, img_size: int | None = None, blocks: int | None = None) -> None:
    """Keep the first `blocks` encoder blocks and fix the input size to `img_size` (both optional).

    Fixing the size resamples the position embedding once with the function the dynamic path calls on
    every image (anomalib switches its antialiasing off), so the outputs do not change. A fixed encoder
    refuses other input sizes instead of resampling the already resampled embedding.
    """
    vit = model.encoder.feature_extractor
    if blocks is not None:
        needed = max(model.target_layers) + 1
        if not needed <= blocks <= len(vit.blocks):
            raise ValueError(f"blocks must be between {needed} and {len(vit.blocks)}, got {blocks}")
        vit.blocks = vit.blocks[:blocks]
    if img_size is not None:
        patch = int(model.encoder.patch_size)
        if img_size <= 0 or img_size % patch != 0:
            raise ValueError(f"img_size must be a positive multiple of {patch}, got {img_size}")
        vit.set_input_size(img_size=(img_size, img_size))
        model.encoder.register_forward_pre_hook(_size_guard(img_size))
        model.fixed_img_size = img_size


# 남아 있는 인코더 블록 수. 테스트가 fix_encoder 뒤에 10블록인지 확인하는 데 쓴다.
def encoder_blocks(model: torch.nn.Module) -> int:
    return len(model.encoder.feature_extractor.blocks)


# ONNX로 내보낼 추론 그래프. 정규화한 이미지 [B, 3, S, S]를 받아 (이미지 점수 [B], 256×256 맵 [B, 256, 256])을
# 낸다. 맵 리사이즈·흐림·상위 1% 평균까지 그래프 안에 넣어(6단계 관문 1 규칙), 서빙
# 쪽(inspector.ReconstructionInspector)은 onnxruntime 호출 한 번으로 점수와 히트맵을 받고 torch·anomalib 없이
# 돈다.
# 점수는 eval 모드 DinomalyModel.forward와 같다. 맵은 흐리기 전의 맵을 256으로 키운 것으로,
# run_dinomaly.predict가 픽셀 지표용으로 저장하는 것과 같다. 내보낸 그래프와 torch의 차이는 점수 상대 8.0e-7,
# 맵 절대 5.1e-7이었다(6단계 test 기록, 허용 1e-3). 호출 위치: dinomaly_serving의 ONNX 내보내기.
class ServingGraph(torch.nn.Module):
    """Normalised image [B, 3, S, S] -> (image score [B], map [B, 256, 256]) of a Dinomaly model.

    The score is that of `DinomalyModel.forward` in eval mode: the mean of the top 1% of the blurred
    256x256 map (top-k instead of a full sort; the same values are averaged). The map is the unblurred
    map resized to 256x256, which is what `run_dinomaly.predict` stores for the pixel metrics.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        # anomalib이 점수를 내는 맵 크기가 256이 아니면(라이브러리 버전이 바뀐 경우) 이 그래프의 점수가
        # anomalib과 달라지므로 만들 때 멈춘다.
        if DEFAULT_RESIZE_SIZE != MAP_SIZE:
            raise RuntimeError(f"anomalib scores on a {DEFAULT_RESIZE_SIZE} map, this graph on {MAP_SIZE}")
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 인코더 특징과 디코더 복원의 차이로 입력 크기의 이상 맵을 만들고 256×256으로 키운다. 그 맵을
        # anomalib의 가우시안 흐림에 통과시켜 펼친 뒤, 상위 1%(DEFAULT_MAX_RATIO) 값의 평균을 이미지 점수로
        # 쓴다. 전체 정렬 대신 topk를 써도 평균하는 값은 같다(docstring). 반환하는 맵은 흐리기 전 맵의 채널
        # 0이다.
        en, de = self.model.get_encoder_decoder_outputs(x)
        amap, _ = self.model.calculate_anomaly_maps(en, de, out_size=(x.shape[2], x.shape[3]))
        amap = F.interpolate(amap, size=(MAP_SIZE, MAP_SIZE), mode="bilinear", align_corners=False)
        flat = self.model.gaussian_blur(amap).flatten(1)
        k = int(flat.shape[1] * DEFAULT_MAX_RATIO)
        score = torch.topk(flat, k, dim=1).values.mean(dim=1)
        return score, amap[:, 0]
