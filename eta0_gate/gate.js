"use strict";

const byId = id => document.getElementById(id);
const token = location.hash.slice(1) || sessionStorage.getItem("eta0_gate");
if (token) sessionStorage.setItem("eta0_gate", token);
history.replaceState(null, "", location.pathname);

const headers = {
  Authorization: "Bearer " + (token || ""),
  "Content-Type": "application/json",
};

const PHASE_NAMES = {
  preparing: "正在准备",
  question: "等待澄清",
  review: "等待确认",
  running: "27B 执行中",
  finished: "已结束",
};

const STEP_ORDER = ["preparing", "question", "review", "running", "finished"];

const REASON_LABELS = {
  native_success: "原生环境条件已满足",
  user_cancelled: "用户在执行前取消了请求",
  user_stopped: "用户在运行中手动停止了执行",
  action_budget: "已达动作步数预算上限",
  call_budget: "已达模型调用预算上限",
  wall_time_budget: "已达单轮时间预算上限",
  needs_clarification: "缺少必要澄清回答，未能开跑",
  preparation_error: "上层准备阶段发生错误",
  runtime_error: "运行期发生异常中断",
};

let state = {};
let sending = false;
let sawQuestion = false;
let pollErrors = 0;
let timerHandle = null;
let runStartLocal = null;

function showNotice(text, tone = "info") {
  const el = byId("notice");
  if (!text) {
    el.hidden = true;
    el.textContent = "";
    delete el.dataset.tone;
    return;
  }
  el.textContent = text;
  el.dataset.tone = tone;
  el.hidden = false;
}

function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "-";
  if (seconds < 60) return seconds.toFixed(1) + "s";
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return mins + "m " + String(secs).padStart(2, "0") + "s";
}

function formatClock(totalSeconds) {
  const clamped = Math.max(0, Math.floor(totalSeconds));
  const mins = Math.floor(clamped / 60);
  const secs = clamped % 60;
  return String(mins).padStart(2, "0") + ":" + String(secs).padStart(2, "0");
}

function stopTimer() {
  if (timerHandle !== null) {
    clearInterval(timerHandle);
    timerHandle = null;
  }
}

function startTimer(runningSince) {
  stopTimer();
  const nowSec = Date.now() / 1000;
  const skewed = typeof runningSince !== "number" || Math.abs(nowSec - runningSince) > 3600;
  runStartLocal = skewed ? Date.now() : runningSince * 1000;
  const tick = () => {
    byId("elapsedTimer").textContent = formatClock((Date.now() - runStartLocal) / 1000);
  };
  tick();
  timerHandle = setInterval(tick, 1000);
}

function updateStepper(phase, hasDialogue) {
  if (phase === "question" || hasDialogue) sawQuestion = true;
  const activeIdx = STEP_ORDER.indexOf(phase);
  document.querySelectorAll(".step").forEach(el => {
    const key = el.dataset.step;
    const idx = STEP_ORDER.indexOf(key);
    let stepState = "pending";
    if (key === "question" && !sawQuestion && activeIdx > 1) {
      stepState = "skipped";
    } else if (idx < activeIdx) {
      stepState = "done";
    } else if (idx === activeIdx) {
      stepState = "active";
    }
    el.dataset.state = stepState;
  });
}

function renderDialogue(dialogue) {
  const wrap = byId("dialogueWrap");
  const list = byId("dialogue");
  const count = byId("dialogueCount");
  const rows = Array.isArray(dialogue) ? dialogue : [];
  list.textContent = "";
  wrap.hidden = rows.length === 0;
  if (!rows.length) return;
  count.textContent = rows.length + " 条记录";

  for (const item of rows) {
    const card = document.createElement("div");
    card.className = "turn-card";

    const qRow = document.createElement("div");
    qRow.className = "turn-q";
    const qTag = document.createElement("span");
    qTag.className = "turn-tag turn-tag-q";
    qTag.textContent = "问";
    const qText = document.createElement("span");
    qText.textContent = item.question || "";
    qRow.append(qTag, qText);

    const aRow = document.createElement("div");
    aRow.className = "turn-a";
    const aTag = document.createElement("span");
    aTag.className = "turn-tag turn-tag-a";
    aTag.textContent = "答";
    const aText = document.createElement("span");
    aText.textContent = item.answer || "";
    aRow.append(aTag, aText);

    card.append(qRow, aRow);
    list.append(card);
  }
}

function renderTelemetry(s) {
  const hasObs = Boolean(s.observation_id);
  const hasFrame = Boolean(s.reference_frame);
  const hasGripper = typeof s.gripper_open_width_mm === "number";
  byId("teleObsWrap").hidden = !hasObs;
  byId("teleFrameWrap").hidden = !hasFrame;
  byId("teleGripperWrap").hidden = !hasGripper;
  if (hasObs) byId("teleObs").textContent = s.observation_id;
  if (hasFrame) byId("teleFrame").textContent = s.reference_frame;
  if (hasGripper) byId("teleGripper").textContent = s.gripper_open_width_mm.toFixed(1) + " mm";
  byId("telemetry").hidden = !(hasObs || hasFrame || hasGripper);
}

function updateDiffState() {
  const current = byId("instruction").value.trim();
  const goal = (state.goal || "").trim();
  const proposed = (state.proposed_instruction || state.instruction || "").trim();
  const badge = byId("diffBadge");
  const restoreProposed = byId("restoreProposed");

  if (current === goal) {
    badge.textContent = "与原话一致";
    badge.className = "chip chip-neutral";
  } else if (current === proposed && proposed !== goal) {
    badge.textContent = "上层已改写";
    badge.className = "chip chip-teal";
  } else {
    badge.textContent = "已手动编辑";
    badge.className = "chip chip-amber";
  }

  restoreProposed.hidden = !(proposed && proposed !== goal && current !== proposed);
  byId("instructionCount").textContent = byId("instruction").value.length + " / 1500";
}

function updateDemoToggleUI() {
  const checked = byId("keep").checked;
  byId("demoBlock").dataset.enabled = checked ? "true" : "false";
  byId("keepLabel").textContent = checked ? "附带此示例" : "不使用示例";
}

function renderDemo(demo) {
  const block = byId("demoBlock");
  block.hidden = !demo;
  byId("keep").checked = true;
  updateDemoToggleUI();
  if (!demo) return;

  byId("demoId").textContent = demo.id || "";
  byId("demoGoal").textContent = demo.goal || "";
  const previewBtn = byId("demoPreviewBtn");
  previewBtn.hidden = !demo.preview;
  if (demo.preview) byId("preview").src = demo.preview;

  const meta = byId("demoMeta");
  meta.textContent = "";
  const tags = [];
  if (demo.suite) tags.push(demo.suite);
  if (Array.isArray(demo.segments) && demo.segments.length) {
    tags.push("采样 " + demo.segments.length + " 个动作段");
  }
  if (demo.frames_count) tags.push(demo.frames_count + " 帧参考图");
  for (const label of tags) {
    const chip = document.createElement("span");
    chip.className = "chip chip-muted";
    chip.textContent = label;
    meta.append(chip);
  }
}

function renderFinished(s) {
  const banner = byId("verdictBanner");
  const icon = byId("verdictIcon");
  const outcome = byId("outcome");
  const detail = byId("outcomeDetail");
  const metrics = byId("metricsGrid");

  let tone = "fail";
  let symbol = "!";
  let title = "环境目标未达成";
  let sub = REASON_LABELS[s.reason] || s.reason || "";

  if (s.reason === "user_cancelled") {
    tone = "cancelled";
    symbol = "×";
    title = "已取消本次请求";
  } else if (s.reason === "user_stopped") {
    tone = "cancelled";
    symbol = "■";
    title = "已手动停止执行";
  } else if (s.reason === "runtime_error") {
    tone = "error";
    symbol = "!";
    title = "运行中断";
    sub = s.error_type ? "异常类型: " + s.error_type : sub;
  } else if (s.success) {
    tone = "success";
    symbol = "✓";
    title = "环境目标已达成";
  }

  document.body.dataset.outcome = tone;
  banner.dataset.tone = tone;
  icon.textContent = symbol;
  outcome.textContent = title;
  detail.textContent = s.reason ? s.reason + (sub ? " · " + sub : "") : sub;

  const showMetrics = s.reason !== "runtime_error";
  metrics.hidden = !showMetrics;
  if (showMetrics) {
    byId("metricAttempts").textContent = String(s.attempts ?? (s.calls ? 1 : 0));
    byId("metricActions").textContent = String(s.actions ?? 0);
    const lowerCalls = s.calls ?? 0;
    const upperCalls = s.upper_calls ?? 0;
    byId("metricCalls").textContent = upperCalls > 0 ? lowerCalls + " + " + upperCalls : String(lowerCalls);
    byId("metricTime").textContent = formatDuration(s.elapsed_s);
  }

  const hasReceipt = Boolean(s.receipt);
  byId("path").textContent = hasReceipt ? s.receipt : "未生成回执，请查看启动终端中的日志记录";
  byId("copyPath").hidden = !hasReceipt;
}

function setButtonsDisabled(disabled) {
  for (const id of ["answerSubmit", "startSubmit", "cancel"]) {
    const el = byId(id);
    if (el) el.disabled = disabled;
  }
}

function render(s) {
  if (state.version === s.version) return;
  const prevPhase = state.phase;
  state = s;

  document.body.dataset.phase = s.phase || "preparing";
  if (s.phase !== "finished") delete document.body.dataset.outcome;

  byId("phase").textContent = PHASE_NAMES[s.phase] || s.phase;
  byId("benchmark").textContent = s.benchmark_goal || "";
  byId("goal").textContent = s.goal || "正在等待请求内容…";

  for (const view of ["external", "wrist"]) {
    if (s[view]) {
      byId(view).src = s[view];
      byId(view + "Box").dataset.loading = "false";
    }
  }
  const hasViews = Boolean(s.external && s.wrist);
  if (s.phase === "running" && s.live_frame) {
    byId("captureBadge").textContent = "实时画面 · 动作 " + (s.live_actions ?? 0);
    byId("captureBadge").className = "chip chip-teal";
  } else {
    byId("captureBadge").textContent = hasViews ? "双视角已就绪" : "等待采集";
    byId("captureBadge").className = hasViews ? "chip chip-teal" : "chip chip-muted";
  }

  renderTelemetry(s);
  renderDialogue(s.dialogue);
  updateStepper(s.phase, Array.isArray(s.dialogue) && s.dialogue.length > 0);

  const isPreparing = s.phase === "preparing";
  const isQuestion = s.phase === "question";
  const isReview = s.phase === "review";
  const isRunning = s.phase === "running";
  const isFinished = s.phase === "finished";

  byId("preparingPanel").hidden = !isPreparing;
  byId("questionForm").hidden = !isQuestion;
  byId("reviewForm").hidden = !isReview;
  byId("runningPanel").hidden = !isRunning;
  byId("receipt").hidden = !isFinished;
  byId("cancelWrap").hidden = !(isQuestion || isReview);

  if (isPreparing && prevPhase && prevPhase !== "preparing") {
    byId("preparingTitle").textContent = "正在更新交接方案";
    byId("preparingDesc").textContent = "已收到您的输入，正在继续准备下一步交接内容…";
  }

  if (isQuestion) {
    byId("question").textContent = s.question || "";
    byId("answer").value = "";
    byId("answerCount").textContent = "0 / 1500";
    byId("answer").focus();
  }

  if (isReview) {
    byId("instruction").value = s.instruction || "";
    renderDemo(s.demo);
    updateDiffState();
  }

  if (isRunning) {
    byId("runningAttempt").textContent = s.stopping
      ? "正在安全停止…"
      : "第 " + (s.attempt || 1) + " 轮自主执行中";
    byId("runningInstruction").textContent = s.instruction || "";
    byId("liveActions").textContent = String(s.live_actions ?? 0);
    byId("liveCalls").textContent = String(s.live_calls ?? 0);
    const stopBtn = byId("stopRun");
    stopBtn.disabled = Boolean(s.stopping);
    stopBtn.textContent = s.stopping ? "正在等待当前动作段结束…" : "安全停止执行";
    const demoTag = byId("runningDemoTag");
    if (s.demo_id) {
      demoTag.textContent = "携带示例 · " + s.demo_id;
      demoTag.className = "chip chip-teal";
    } else {
      demoTag.textContent = "零示例执行";
      demoTag.className = "chip chip-muted";
    }
    if (prevPhase !== "running" || timerHandle === null) startTimer(s.running_since);
  } else {
    stopTimer();
  }

  if (isFinished) {
    renderFinished(s);
    showNotice("");
  }

  setButtonsDisabled(false);
}

async function poll() {
  try {
    const requestHeaders = { ...headers };
    if (state.version !== undefined) requestHeaders["If-None-Match"] = '"' + state.version + '"';
    const r = await fetch("/state", { headers: requestHeaders, cache: "no-store" });
    if (r.status === 304) {
      pollErrors = 0;
      if (state.phase !== "finished") setTimeout(poll, 750);
      return;
    }
    if (r.status === 403) {
      showNotice(
        "会话令牌缺失或已失效（403）：请复制并打开终端中最新打印的完整链接（包含 # 后的令牌）。",
        "warn",
      );
      return;
    }
    if (!r.ok) throw Error("HTTP " + r.status);
    pollErrors = 0;
    const data = await r.json();
    if (!sending) showNotice("");
    render(data);
  } catch {
    pollErrors += 1;
    if (state.phase !== "finished" && pollErrors >= 2) {
      showNotice("暂时无法连接到 Gate 服务，请确认启动终端中的会话进程与端口转发状态。", "warn");
    }
  }
  if (state.phase !== "finished") setTimeout(poll, 750);
}

async function submit(data) {
  if (sending) return;
  sending = true;
  setButtonsDisabled(true);
  try {
    const r = await fetch("/decision", {
      method: "POST",
      headers,
      body: JSON.stringify({ ...data, version: state.version }),
    });
    if (!r.ok) throw Error("HTTP " + r.status);
    showNotice("已提交，正在同步至执行器…", "info");
  } catch {
    showNotice("交接状态已更新或提交失败，请检查当前界面状态后重试。", "error");
    setButtonsDisabled(false);
  } finally {
    sending = false;
  }
}

function openLightbox(src, caption) {
  if (!src) return;
  byId("lightboxImg").src = src;
  byId("lightboxCaption").textContent = caption;
  byId("lightbox").hidden = false;
  byId("lightboxClose").focus();
}

function closeLightbox() {
  byId("lightbox").hidden = true;
  byId("lightboxImg").removeAttribute("src");
}

/* Event Wiring */
byId("restore").onclick = () => {
  byId("instruction").value = state.goal || "";
  updateDiffState();
  byId("instruction").focus();
};

byId("restoreProposed").onclick = () => {
  byId("instruction").value = state.proposed_instruction || state.instruction || "";
  updateDiffState();
  byId("instruction").focus();
};

byId("instruction").addEventListener("input", updateDiffState);
byId("keep").addEventListener("change", updateDemoToggleUI);

byId("answer").addEventListener("input", () => {
  byId("answerCount").textContent = byId("answer").value.length + " / 1500";
});

byId("answer").addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    byId("questionForm").requestSubmit();
  }
});

byId("instruction").addEventListener("keydown", e => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    byId("reviewForm").requestSubmit();
  }
});

byId("cancel").onclick = () => submit({ action: "cancel" });
byId("stopRun").onclick = () => submit({ action: "stop" });

byId("questionForm").onsubmit = e => {
  e.preventDefault();
  const value = byId("answer").value.trim();
  if (!value) return;
  submit({ action: "answer", answer: value });
};

byId("reviewForm").onsubmit = e => {
  e.preventDefault();
  const value = byId("instruction").value.trim();
  if (!value) return;
  submit({
    action: "start",
    instruction: value,
    keep_demo: byId("keep").checked,
  });
};

document.querySelectorAll(".zoom-trigger").forEach(btn => {
  btn.addEventListener("click", () => {
    const key = btn.dataset.zoom;
    if (!state[key]) return;
    const caption = key === "external" ? "CAM-01 · 外部全局视角" : "CAM-02 · 腕部夹爪视角";
    openLightbox(state[key], caption);
  });
});

byId("demoPreviewBtn").addEventListener("click", () => {
  if (state.demo && state.demo.preview) {
    openLightbox(state.demo.preview, "成功轨迹示例 · " + (state.demo.id || ""));
  }
});

byId("lightboxClose").addEventListener("click", closeLightbox);
byId("lightbox").addEventListener("click", e => {
  if (e.target === byId("lightbox")) closeLightbox();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && !byId("lightbox").hidden) closeLightbox();
});

byId("copyPath").addEventListener("click", async () => {
  const text = byId("path").textContent || "";
  if (!text) return;
  const btn = byId("copyPath");
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "已复制 ✓";
    setTimeout(() => {
      btn.textContent = "复制路径";
    }, 1800);
  } catch {
    btn.textContent = "请手动选择复制";
    setTimeout(() => {
      btn.textContent = "复制路径";
    }, 1800);
  }
});

poll();
