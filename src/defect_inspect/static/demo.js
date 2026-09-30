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
  if (!state.photo) return;
  for (const id of ["photo", "overlay"]) {
    const canvas = $(id);
    const ctx = canvas.getContext("2d");
    ctx.globalAlpha = 1;
    ctx.drawImage(state.photo, 0, 0, canvas.width, canvas.height);
  }
  if (!state.heat) return;
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

async function inspect(event) {
  event.preventDefault();
  const file = $("image").files[0];
  if (!file) return;
  const body = new FormData();
  body.append("image", file);
  body.append("category", $("category").value);
  $("submit").disabled = true;
  setStatus("검사 중…", false);
  try {
    const response = await fetch("/inspect", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) {
      setStatus(typeof data.detail === "string" ? data.detail : "요청이 거부됐다.", true);
      return;
    }
    const url = URL.createObjectURL(file);
    state.photo = await loadImage(url);
    URL.revokeObjectURL(url);
    state.heat = data.heatmap_png ? await loadImage("data:image/png;base64," + data.heatmap_png) : null;
    const verdict = $("verdict");
    verdict.textContent = data.is_defect ? "불량" : "양품";
    verdict.className = "badge " + (data.is_defect ? "bad" : "ok");
    $("score").textContent = data.score.toFixed(3);
    $("threshold").textContent = data.threshold.toFixed(3);
    $("latency").textContent = String(data.latency_ms);
    $("result").hidden = false;
    draw();
    setStatus("", false);
  } catch (err) {
    setStatus("검사에 실패했다. 서버 상태를 확인한다.", true);
  } finally {
    $("submit").disabled = false;
  }
}

$("form").addEventListener("submit", inspect);
$("opacity").addEventListener("input", draw);
loadCategories();
