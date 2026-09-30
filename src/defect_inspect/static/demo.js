"use strict";

const $ = (id) => document.getElementById(id);
const state = { photo: null, heat: null };

function setStatus(text, isError) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", Boolean(isError));
}

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

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

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
  const buffer = document.createElement("canvas");
  buffer.width = state.heat.width;
  buffer.height = state.heat.height;
  const bctx = buffer.getContext("2d");
  bctx.drawImage(state.heat, 0, 0);
  const pixels = bctx.getImageData(0, 0, buffer.width, buffer.height);
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

function describeRefusal(data) {
  return data && typeof data.detail === "string" ? data.detail : "요청이 거부됐다.";
}

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
  } finally {
    $("submit").disabled = false;
  }

  const verdict = $("verdict");
  verdict.textContent = data.is_defect ? "불량" : "양품";
  verdict.className = "badge " + (data.is_defect ? "bad" : "ok");
  $("score").textContent = data.score.toFixed(3);
  $("threshold").textContent = data.threshold.toFixed(3);
  $("latency").textContent = String(data.latency_ms);
  $("result").hidden = false;
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

$("form").addEventListener("submit", inspect);
$("opacity").addEventListener("input", draw);
loadCategories();
