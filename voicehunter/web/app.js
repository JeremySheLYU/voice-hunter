"use strict";
const el = (id) => document.getElementById(id);
const player = el("player");
const state = {
  source: null, separationId: null, renderId: null, mix: null, listening: "original",
  busy: false, modes: [], peaks: [], seekAfterLoad: null, stopAt: null,
  dirtyMix: false, canVideo: false, renderKey: null, modePan: {stereo:0,headphones:0,"headphones-measured":0}, outputMode: "stereo",
};

function message(text = "", error = false) {
  const box = el("notice");
  box.textContent = text;
  box.hidden = !text;
  box.classList.toggle("error", error);
}
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || data.message || `伺服器回應 ${response.status}`);
  return data;
}
const post = (url, data) => api(url, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data),
});
const clock = (seconds, decimal = false) => {
  const value = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  return Math.floor(value / 60) + ":" + Math.floor(value % 60).toString().padStart(2, "0")
    + (decimal ? "." + Math.floor(value * 10 + 0.00001) % 10 : "");
};
const activeMode = () => state.modes.find((m) => m.id === el("mode").value);
const currentUrl = () => {
  if (state.listening === "mix") return state.mix;
  if (state.listening === "target" || state.listening === "residual") return el("dl-" + state.listening).href;
  return state.source?.original_url;
};
function controls() {
  el("file").disabled = state.busy;
  el("demo").disabled = el("duet-demo").disabled = el("iphone-demo").disabled = state.busy;
  el("separation-settings").disabled = !state.source || !activeMode() || state.busy;
  el("mix-settings").disabled = !state.separationId || state.busy;
  el("play").disabled = el("listen-original").disabled = el("seek").disabled = !state.source || state.busy;
  el("listen-mix").disabled = !state.mix || state.busy;
  el("export-video").disabled = state.busy;
  el("stale-mix").hidden = !state.dirtyMix;
  el("render").textContent = state.mix && state.dirtyMix ? "套用新設定" : "套用混音";
  for (const key of ["target", "residual"]) {
    const button = el("solo-" + key);
    button.disabled = !state.separationId || state.busy;
    button.setAttribute("aria-pressed", String(state.listening === key));
  }
  for (const key of ["original", "mix"]) {
    const btn = el("listen-" + key);
    const selected = state.listening === key;
    btn.classList.toggle("active", selected);
    btn.setAttribute("aria-pressed", String(selected));
  }
}
async function task(status, action) {
  if (state.busy) return;
  state.busy = true;
  player.pause(); state.stopAt = null;
  controls(); message(status);
  try { await action(); }
  catch (error) { message(error.message || "操作失敗，請重試。", true); }
  finally { state.busy = false; controls(); }
}
function paint() {
  const canvas = el("wave");
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  const scale = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(w * scale) || canvas.height !== Math.round(h * scale)) {
    canvas.width = Math.round(w * scale); canvas.height = Math.round(h * scale);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#98a77c"; ctx.lineWidth = 0.55;
  for (let i = 0; i <= 20; i++) {
    const x = i * w / 20; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (const y of [h / 4, h / 2, 3 * h / 4]) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }
  if (!state.source) return;
  const duration = state.source.duration;
  if (activeMode()?.kind === "example") {
    const ranges = [["start", "end", "rgba(231,144,47,.36)"]];
    if (el("use-background").checked) ranges.push(["bg-start", "bg-end", "rgba(73,120,156,.31)"]);
    for (const [a,b,color] of ranges) {
      const left = el(a).valueAsNumber / duration * w, right = el(b).valueAsNumber / duration * w;
      if (right > left) { ctx.fillStyle = color; ctx.fillRect(left, 0, right - left, h); }
    }
  }
  if (state.peaks.length) {
    const step = w / state.peaks.length;
    ctx.fillStyle = "#35452e";
    state.peaks.forEach((peak, i) => {
      const amp = Math.min(1, Math.max(0, Number(peak))) * h * 0.45;
      ctx.fillRect(i * step, h / 2 - amp, Math.max(1, step * 0.72), Math.max(1, amp * 2));
    });
  }
  const pos = Math.min(w - 2, Math.max(0, Number(el("seek").value) / duration * w));
  ctx.fillStyle = "#672e13"; ctx.fillRect(pos, 0, 2, h);
}
function showPosition(t) {
  el("seek").value = t;
  el("time").textContent = clock(t, true) + " / " + clock(state.source?.duration, true);
  el("seek").setAttribute("aria-valuetext", clock(t, true));
  paint();
}
function changeMonitor(key, time, play) {
  if (!state.source || (key === "mix" && !state.mix)) return;
  state.listening = key;
  const url = new URL(currentUrl(), location.href).href;
  const t = Math.min(Math.max(0, time), state.source.duration);
  if (player.src !== url) {
    state.seekAfterLoad = t;
    player.src = url;
    player.load();
  } else if (player.readyState >= 1) {
    state.seekAfterLoad = null; player.currentTime = t;
  } else state.seekAfterLoad = t;
  if (play) {
    player.play().catch((error) => {
      if (error.name !== "AbortError") message("無法播放音軌。可以下載 WAV 後試聽。", true);
    });
  }
  showPosition(t); controls();
}
player.addEventListener("loadedmetadata", () => {
  if (state.seekAfterLoad !== null) {
    player.currentTime = Math.min(state.seekAfterLoad, player.duration || state.seekAfterLoad);
    state.seekAfterLoad = null;
  }
});
player.addEventListener("timeupdate", () => {
  if (state.stopAt !== null && player.currentTime >= state.stopAt) {
    const end = state.stopAt; state.stopAt = null; player.pause(); player.currentTime = end;
  }
  if (state.seekAfterLoad === null) showPosition(player.currentTime);
});
function playLabel() {
  el("play").innerHTML = player.paused ? '<span aria-hidden="true">▶</span> 播放' : '<span aria-hidden="true">Ⅱ</span> 暫停';
}
player.addEventListener("play", playLabel);
player.addEventListener("pause", playLabel);
player.addEventListener("ended", () => { state.stopAt = null; playLabel(); });
el("play").onclick = () => {
  if (!player.paused) { player.pause(); return; }
  state.stopAt = null;
  changeMonitor(state.listening, player.ended ? 0 : Number(el("seek").value), true);
};
el("seek").oninput = () => {
  state.stopAt = null;
  const t = Number(el("seek").value);
  if (player.readyState >= 1) { state.seekAfterLoad = null; player.currentTime = t; }
  else state.seekAfterLoad = t;
  showPosition(t);
};
["original", "mix"].forEach((key) => {
  el("listen-" + key).onclick = () => {
    state.stopAt = null; changeMonitor(key, Number(el("seek").value), !player.paused);
  };
});
["target", "residual"].forEach((key) => {
  el("solo-" + key).onclick = () => {
    if (state.listening === key && !player.paused) { player.pause(); return; }
    state.stopAt = null; changeMonitor(key, player.ended ? 0 : Number(el("seek").value), true);
  };
});
function clearOutputs() {
  state.separationId = state.renderId = state.mix = state.renderKey = null;
  state.dirtyMix = false;
  el("stems").hidden = el("exports").hidden = true;
  el("dl-video").hidden = el("export-video").hidden = true;
  for (const id of ["dl-target", "dl-residual", "dl-mix", "dl-video"]) el(id).removeAttribute("href");
}
function acceptSource(data) {
  if (!data.id || !data.original_url || !(data.duration > 0)) throw new Error("匯入資料缺少可播放的音軌。");
  const duration = Number(data.duration);
  clearOutputs();
  state.source = { ...data, duration };
  state.canVideo = !!data.has_video;
  state.peaks = Array.isArray(data.peaks) ? data.peaks : [];
  el("filename").textContent = data.name || data.filename || "音訊";
  const desc = [];
  if (data.channels) desc.push(Number(data.channels) === 2 ? "Stereo" : data.channels + " ch");
  if (data.sample_rate) desc.push((data.sample_rate / 1000).toFixed(1).replace(/\.0$/, "") + " kHz");
  el("track-meta").textContent = desc.join(" · ") || "音訊";
  el("wave-empty").hidden = true; el("seek").max = duration;
  el("axis-mid").textContent = clock(duration / 2); el("axis-end").textContent = clock(duration);
  const suggestion = data.suggested || data.suggestions || data;
  const fmt = (v) => Math.round(Number(v) * 1000) / 1000;
  [["start",suggestion.start ?? 0],["end",suggestion.end ?? Math.min(2,duration)],
    ["bg-start",suggestion.background_start ?? Math.max(0,duration - 2)],
    ["bg-end",suggestion.background_end ?? duration]].forEach(([id,v]) => {
    el(id).max = duration; el(id).value = fmt(v);
  });
  if (suggestion.mode && state.modes.some((m) => m.id === suggestion.mode)) el("mode").value = suggestion.mode;
  el("use-background").checked = suggestion.background_start != null && suggestion.background_end != null;
  modeUI();
  state.outputMode = "stereo"; el("output-mode").value = "stereo"; updateOutputMode(); resetSliders(); changeMonitor("original", 0, false); message(data.note || "");
}
function readUpload(file) {
  if (!file) return;
  task("正在匯入並讀取聲道…", async () => {
    const data = await api("/api/import", {method:"POST", headers:{
      "Content-Type":"application/octet-stream","X-Filename":encodeURIComponent(file.name),
    }, body:file});
    acceptSource(data);
  });
}
el("file").onchange = () => {readUpload(el("file").files[0]); el("file").value="";};
const zone = el("session");
for (const name of ["dragenter","dragover"]) zone.addEventListener(name, (event) => {
  event.preventDefault(); if (!state.busy) zone.classList.add("dragging");
});
zone.addEventListener("dragleave", (event) => { if (!zone.contains(event.relatedTarget)) zone.classList.remove("dragging"); });
zone.addEventListener("drop", (event) => {
  event.preventDefault(); zone.classList.remove("dragging"); readUpload(event.dataTransfer.files[0]);
});
function demo(which) {
  return task("正在開啟試聽素材…", async () => {
    const data = await api(which === "studio" ? "/api/demo" : "/api/demo?clip=" + encodeURIComponent(which));
    const preferred = {iphone:"rnnoise",duet:"speakers",studio:"example"}[which];
    if (state.modes.some((m) => m.id === preferred)) el("mode").value = preferred;
    acceptSource(data);
  });
}
el("demo").onclick = () => demo("studio"); el("duet-demo").onclick = () => demo("duet"); el("iphone-demo").onclick = () => demo("iphone");
function modeUI() {
  const mode = activeMode(), kind = mode?.kind || "example";
  el("example-settings").hidden = kind !== "example";
  el("target-text-row").hidden = kind !== "text";
  el("target-choice-row").hidden = kind !== "choice";
  const list = mode?.options || [];
  el("target-choice").replaceChildren(...list.map((v) => typeof v === "string" ? new Option(v,v) : new Option(v.label,v.value)));
  const required = mode?.id === "speakers" || mode?.requires_background;
  if (required) el("use-background").checked = true;
  el("use-background").disabled = !!required;
  el("sample-instructions").textContent = required ?
    "分別選出目標 A 與聲源 B 各自清楚出現的時段。兩者的左右方位有差別時最適用。" :
    "選一段盡量只有目標聲音的時間。若範例裡有人聲和音樂，模型可能把兩者一起提取。";
  el("audition").textContent = required ? "試聽 A" : "試聽選段";
  el("audition-bg").textContent = required ? "試聽 B" : "試聽背景";
  el("ref-label").textContent = required ? "第二個聲源的單獨選段" : "加入只有背景聲的對照選段";
  el("background-settings").hidden = !el("use-background").checked;
  el("method-note").textContent = mode?.note || mode?.description || "";
  el("method-note").hidden = !el("method-note").textContent;
  paint(); controls();
}
function separationChanged() {
  if (!state.separationId) return;
  const position = Number(el("seek").value);
  clearOutputs();
  changeMonitor("original", position, false);
  controls();
  message("分軌條件已更改，請重新提取聲源。");
}
el("mode").onchange = () => { separationChanged(); modeUI(); };
el("target-choice").onchange = separationChanged;
el("target-text").addEventListener("input",separationChanged);
el("use-background").onchange = () => { separationChanged(); el("background-settings").hidden = !el("use-background").checked; paint(); };
for (const id of ["start","end","bg-start","bg-end"]) el(id).addEventListener("input",() => { separationChanged(); paint(); });
function interval(background) {
  const a = el(background ? "bg-start" : "start").valueAsNumber;
  const b = el(background ? "bg-end" : "end").valueAsNumber;
  if (!Number.isFinite(a) || !Number.isFinite(b) || a < 0 || b <= a || b > state.source.duration + 0.001) {
    throw new Error("請輸入有效選段，結束時間須大於開始時間，並落在錄音範圍內。");
  }
  return [a, Math.min(b,state.source.duration)];
}
function audition(background) {
  try { const [a,b] = interval(background); state.stopAt = b; changeMonitor("original", a, true); }
  catch (error) { message(error.message, true); }
}
el("audition").onclick = () => audition(false); el("audition-bg").onclick = () => audition(true);
function link(id, url, suffix) {
  const a = el(id); a.href = url;
  a.download = (state.source.name || "VoiceHunter").replace(/\.[^.]+$/,"") + suffix;
  a.hidden = false;
}
el("separate").onclick = () => task("正在提取聲源…", async () => {
  const mode = activeMode();
  const params = { id:state.source.id, mode:mode.id };
  if (mode.kind === "example") {
    [params.start,params.end] = interval(false);
    if (el("use-background").checked) {
      [params.background_start,params.background_end] = interval(true);
      if (params.start < params.background_end && params.background_start < params.end) throw new Error("目標與對照選段須分開。");
    }
  } else if (mode.kind === "choice") params.target = el("target-choice").value;
  else if (mode.kind === "text") {
    params.target = el("target-text").value.trim();
    if (!params.target) throw new Error("請輸入要提取的聲音。");
  }
  const data = await post("/api/separate",params);
  if (!data.separation_id || !data.target_url || !data.residual_url) throw new Error("伺服器未傳回完整分軌。");
  clearOutputs(); state.separationId = data.separation_id;
  changeMonitor("original", Number(el("seek").value), false);
  link("dl-target", data.target_url, "-target.wav"); link("dl-residual", data.residual_url, "-residual.wav");
  el("stems").hidden = false;
  message(data.note || "聲源已提取，可以調整混音或下載個別音軌。");
});
function mixParams() {
  const knobs = {width:Number(el("width").value),gain_db:Number(el("gain").value)};
  if (state.outputMode !== "stereo") return {...knobs, output_mode:state.outputMode, azimuth_deg:Number(el("pan").value)};
  return {...knobs,pan:Number(el("pan").value)};
}
function syncSliders() {
  const {pan,width,gain_db} = mixParams();
  const headphone = state.outputMode !== "stereo";
  el("pan-val").textContent = headphone ? (Number(el("pan").value) > 0 ? "+" : "") + el("pan").value + "°" : pan ? (pan < 0 ? "左 " : "右 ") + Math.round(Math.abs(pan)*100) + "%" : "原位";
  el("gain-val").textContent = (gain_db > 0 ? "+" : "") + gain_db + " dB";
  el("width-val").textContent = Math.round(width*100) + "%";
  state.dirtyMix = !!state.mix && JSON.stringify(mixParams()) !== state.renderKey;
  controls();
}
function updateOutputMode() {
  const headphone = state.outputMode !== "stereo";
  el("pan").min = headphone ? -90 : -1; el("pan").max = headphone ? 90 : 1;
  el("pan").step = headphone ? 1 : 0.01;
  el("pan").value = state.modePan[state.outputMode];
  el("pan-label").textContent = headphone ? "耳機方位（正前方 0°）" : "目標聲像偏移";
  el("headphone-note").hidden = !headphone;
  el("headphone-note").textContent = state.outputMode === "headphones-measured"
    ? "使用公開實測資料校準的雙耳模型。0° 是正前方；80° 至 90° 沿用最外側的實測角度。"
    : "將目標放在耳機前方的水平聲場。0° 會渲染為正前方。";
}
el("output-mode").onchange = () => {
  state.modePan[state.outputMode] = Number(el("pan").value);
  state.outputMode = el("output-mode").value;
  updateOutputMode(); syncSliders();
};
function resetSliders() {
  state.modePan = {stereo:0,headphones:0,"headphones-measured":0};
  el("pan").value=0; el("width").value=1; el("gain").value=0; syncSliders();
}
for (const id of ["pan","width","gain"]) el(id).addEventListener("input",syncSliders);
el("reset-mix").onclick = resetSliders;
el("wide-mix").onclick = () => {
  el("width").value = 1.35;
  syncSliders();
  el("render").click();
};
el("render").onclick = () => task("正在輸出混音…", async () => {
  const params = mixParams();
  const data = await post("/api/render",{id:state.source.id,separation_id:state.separationId,...params});
  if (!data.mix_url) throw new Error("伺服器未傳回混音。");
  state.mix = data.mix_url; state.renderId = data.render_id; state.renderKey = JSON.stringify(params); state.dirtyMix = false;
  link("dl-mix", data.mix_url, "-mix.wav"); el("exports").hidden = false; el("dl-video").hidden=true;
  el("export-video").hidden = !(state.canVideo && state.renderId);
  if (data.video_url) { link("dl-video",data.video_url,"-mix.mp4"); el("export-video").hidden=true; }
  changeMonitor("mix", Number(el("seek").value), false);
  message(data.note || "混音已完成。播放時可在 A 原音、B 混音之間切換。");
});
el("export-video").onclick = () => task("正在合併原片及混音…", async () => {
  const data = await post("/api/video",{id:state.source.id,render_id:state.renderId});
  if (!data.video_url) throw new Error("伺服器未傳回影片。");
  link("dl-video", data.video_url, data.video_url.toLowerCase().includes(".mov") ? "-mix.mov" : "-mix.mp4");
  el("export-video").hidden = true; message("混音影片已可下載。");
});
new ResizeObserver(paint).observe(el("wavebox"));
(async function init() {
  controls(); paint();
  try {
    const caps = await api("/api/capabilities");
    state.modes = (caps.modes || []).filter((m) => m.available !== false);
    el("mode").replaceChildren(...state.modes.map((m) => new Option(m.label,m.id)));
    el("iphone-demo").hidden = !caps.iphone_demo;
    if (new URLSearchParams(location.search).get("mode") === "five-events" && state.modes.some((m) => m.id === "five-events")) {
      el("mode").value = "five-events";
    }
    el("output-mode-row").hidden = !caps.headphones;
    modeUI();
    const requested = new URLSearchParams(location.search).get("demo");
    if (requested === "studio" || requested === "duet" || (requested === "iphone" && caps.iphone_demo)) await demo(requested);
  } catch { message("無法連上本機工作站，請確認伺服器仍在執行後重新載入。", true); }
})();
