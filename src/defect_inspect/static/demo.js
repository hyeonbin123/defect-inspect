"use strict";

// [파일 역할] 데모 페이지(index.html)의 스크립트. 범주 목록을 불러와 선택지를 채우고, 폼 제출을 가로채 POST
// /inspect?preview=true로 사진을 보낸 뒤, 판정·점수·임계값·처리 시간을 표시하고 모델이 본 입력 사진 위에
// 히트맵을 겹쳐 그린다.
// [설계 이유] 서비스의 CSP(script-src 'self')가 인라인 스크립트를 막아 별도 파일로 둔다. 라이브러리 없이 DOM
// API만 쓴다. 서버가 준 문자열은 textContent로만 넣어 HTML로 해석되지 않게 한다.
// [흐름] 페이지 로드 → loadCategories. 제출 → inspect(전송, 결과 표시) → draw(캔버스). 슬라이더 입력 → draw.
// [연결] service.py의 GET /categories, POST /inspect(heatmap_png, input_png는 base64 PNG).

// $: getElementById의 줄임. state: 마지막으로 받은 입력 사진과 히트맵 이미지. 슬라이더를 움직일 때 다시 받지
// 않고 이것으로 다시 그린다.
const $ = (id) => document.getElementById(id);
const state = { photo: null, heat: null };

// 상태 줄에 메시지를 쓰고, 오류면 error 클래스로 색을 바꾼다(demo.css).
function setStatus(text, isError) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", Boolean(isError));
}

// GET /categories로 범주 목록을 받아 select의 선택지를 만든다. 범주 이름은 value와 textContent로만 넣는다.
// 실패하면 상태 줄에 알린다.
async function loadCategories() {
  try {
    const response = await fetch("/categories");
    if (!response.ok) throw new Error(String(response.status));
    const items = await response.json();
    const select = $("category");
    for (const item of items) {
      const option = document.createElement("option");
      option.value = item.category;
      option.textContent = item.category;
      select.appendChild(option);
    }
  } catch (err) {
    setStatus("범주 목록을 불러오지 못했다.", true);
  }
}

// 이미지 로드를 Promise로 감싸 await로 기다릴 수 있게 한다(여기서는 응답의 base64 PNG를 data: URL로 읽음).
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// 두 캔버스를 지우고 입력 사진을 그린 뒤, 히트맵이 있으면 오른쪽 캔버스에 겹친다. 서버의 히트맵은 회색조(0 =
// 점수 0, 흰색 = 임계값의 1.5배 이상)라서, 화소마다 색을 붉은색으로 바꾸고 회색 값을 알파(불투명도)로 옮긴다.
// 그러면 점수가 높은 곳만 진한 빨강으로 보인다. 그 위에 슬라이더 값으로 전체 진하기를 곱한다.
function draw() {
  for (const id of ["photo", "overlay"]) {
    const canvas = $(id);
    const ctx = canvas.getContext("2d");
    ctx.globalAlpha = 1;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (state.photo) ctx.drawImage(state.photo, 0, 0, canvas.width, canvas.height);
  }
  if (!state.photo || !state.heat) return;
  const canvas = $("overlay");
  const ctx = canvas.getContext("2d");
  // Grey heatmap -> red with alpha proportional to the score.
  // 히트맵 원래 크기의 보조 캔버스에서 화소를 바꾼 뒤 표시 캔버스 크기로 늘려 그린다.
  const buffer = document.createElement("canvas");
  buffer.width = state.heat.width;
  buffer.height = state.heat.height;
  const bctx = buffer.getContext("2d");
  bctx.drawImage(state.heat, 0, 0);
  const pixels = bctx.getImageData(0, 0, buffer.width, buffer.height);
  // 화소는 RGBA 4바이트씩이다. 회색이라 R 값이 곧 점수 밝기이고, 그것을 알파로 옮기고 색은 고정 빨강(255, 40,
  // 20)으로 바꾼다.
  for (let i = 0; i < pixels.data.length; i += 4) {
    const value = pixels.data[i];
    pixels.data[i] = 255;
    pixels.data[i + 1] = 40;
    pixels.data[i + 2] = 20;
    pixels.data[i + 3] = value;
  }
  bctx.putImageData(pixels, 0, 0);
  ctx.globalAlpha = Number($("opacity").value) / 100;
  ctx.drawImage(buffer, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1;
}

// 서버가 거절할 때 준 detail 문자열(예: 413 크기 초과, 400 읽을 수 없는 이미지, 503 바쁨)을 그대로 보여 준다.
// detail이 문자열이 아니면(422 검증 오류는 목록) 일반 문구를 쓴다.
function describeRefusal(data) {
  return data && typeof data.detail === "string" ? data.detail : "요청이 거부됐다.";
}

// Four significant digits: PatchCore scores are around 1 to 3, Dinomaly scores around 0.01, so a fixed
// number of decimals would show a Dinomaly score and its threshold as the same number.
function formatScore(value) {
  return String(Number(value.toPrecision(4)));
}

// 폼 제출 처리. 기본 동작(페이지 이동)을 막고 FormData(image, category)로 보낸다. 보내는 동안 버튼을 꺼 같은
// 사진이 두 번 전송되지 않게 한다.
async function inspect(event) {
  event.preventDefault();
  const file = $("image").files[0];
  if (!file) return;
  const body = new FormData();
  body.append("image", file);
  body.append("category", $("category").value);
  $("submit").disabled = true;
  // The verdict and picture of the previous upload must not stay next to this one.
  $("result").hidden = true;
  state.photo = null;
  state.heat = null;
  draw();
  setStatus("검사 중…", false);

  let data;
  try {
    // preview=true: the server sends back the picture the model saw, so the overlay always lines up
    // (formats a browser cannot draw, EXIF rotation that the server does not apply).
    const response = await fetch("/inspect?preview=true", { method: "POST", body });
    data = await response.json();
    if (!response.ok) {
      setStatus(describeRefusal(data), true);
      return;
    }
  } catch (err) {
    setStatus("검사에 실패했다. 서버 상태를 확인한다.", true);
    return;
  // 성공이든 실패든 버튼을 다시 켠다.
  } finally {
    $("submit").disabled = false;
  }

  // 판정 배지: is_defect(점수 > 임계값)면 "불량"(bad), 아니면 "양품"(ok). 점수와 임계값은 유효숫자 4자리로
  // 보인다.
  const verdict = $("verdict");
  verdict.textContent = data.is_defect ? "불량" : "양품";
  verdict.className = "badge " + (data.is_defect ? "bad" : "ok");
  $("score").textContent = formatScore(data.score);
  $("threshold").textContent = formatScore(data.threshold);
  $("latency").textContent = String(data.latency_ms);
  $("result").hidden = false;
  // 판정은 먼저 보여 주고, 그림은 그다음에 그린다. 그림을 못 그려도 판정은 남는다. heatmap_png가
  // 없으면(heatmap=false 요청) 사진만 그린다.
  try {
    const photo = await loadImage("data:image/png;base64," + data.input_png);
    const heat = data.heatmap_png ? await loadImage("data:image/png;base64," + data.heatmap_png) : null;
    state.photo = photo;
    state.heat = heat;
    draw();
    setStatus("", false);
  } catch (err) {
    setStatus("판정은 나왔지만 사진을 그리지 못했다.", true);
  }
}

// 이벤트 연결과 시작 시 범주 목록 불러오기.
$("form").addEventListener("submit", inspect);
$("opacity").addEventListener("input", draw);
loadCategories();
