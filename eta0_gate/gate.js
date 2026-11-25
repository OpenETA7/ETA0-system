"use strict";
const byId = id => document.getElementById(id);
const token = location.hash.slice(1) || sessionStorage.getItem("eta0_gate");
if (token) sessionStorage.setItem("eta0_gate", token);
history.replaceState(null, "", location.pathname);
const headers = {Authorization: "Bearer " + (token || ""), "Content-Type": "application/json"};
const names = {preparing:"正在准备", question:"等待澄清", review:"等待确认", running:"27B 执行中", finished:"已结束"};
let state = {}, sending = false;

function render(s) {
  if (state.version === s.version) return;
  state = s;
  byId("phase").textContent = names[s.phase];
  byId("benchmark").textContent = s.benchmark_goal;
  byId("goal").textContent = s.goal || "";
  for (const view of ["external","wrist"]) if (s[view]) byId(view).src = s[view];
  byId("notice").textContent = s.phase === "running" ? "第 " + s.attempt + " 轮 · " + s.instruction : "";
  byId("questionForm").hidden = s.phase !== "question";
  byId("reviewForm").hidden = s.phase !== "review";
  byId("cancel").hidden = !["question","review"].includes(s.phase);
  if (s.phase === "question") {
    byId("question").textContent = s.question;
    byId("answer").value = "";
    byId("answer").focus();
  }
  if (s.phase === "review") {
    byId("instruction").value = s.instruction;
    byId("dialogue").textContent = s.dialogue.map(d => d.question + "\n" + d.answer).join("\n");
    byId("dialogue").hidden = !s.dialogue.length;
    byId("demoBlock").hidden = !s.demo;
    byId("keep").checked = true;
    if (s.demo) {
      byId("demoGoal").textContent = s.demo.goal;
      byId("preview").hidden = !s.demo.preview;
      if (s.demo.preview) byId("preview").src = s.demo.preview;
    }
  }
  if (s.phase === "finished") {
    byId("receipt").hidden = false;
    byId("outcome").textContent = s.reason === "user_cancelled" ? "已取消" :
      (s.success ? "环境目标已达成" : "环境目标未达成") + " · " + s.reason;
    byId("path").textContent = s.receipt;
  }
}
async function poll() {
  try {
    const r = await fetch("/state", {headers, cache:"no-store"});
    if (!r.ok) throw Error();
    render(await r.json());
  } catch {
    byId("notice").textContent = "暂时无法连接，请检查终端中的 Gate 链接和进程状态。";
  }
  if (state.phase !== "finished") setTimeout(poll, 750);
}
async function submit(data) {
  if (sending) return;
  sending = true;
  try {
    const r = await fetch("/decision", {method:"POST",headers,body:JSON.stringify({...data,version:state.version})});
    if (!r.ok) throw Error();
    byId("notice").textContent = "已收到";
  } catch {
    byId("notice").textContent = "交接已更新或提交失败，请检查当前状态。";
  } finally { sending = false; }
}
byId("restore").onclick = () => {byId("instruction").value = state.goal;};
byId("cancel").onclick = () => submit({action:"cancel"});
byId("questionForm").onsubmit = e => {e.preventDefault();submit({action:"answer",answer:byId("answer").value});};
byId("reviewForm").onsubmit = e => {e.preventDefault();submit({action:"start",instruction:byId("instruction").value,keep_demo:byId("keep").checked});};
poll();
