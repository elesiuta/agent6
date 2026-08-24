"use strict";
const view = document.getElementById('view');
const crumb = document.getElementById('crumb');
let live = null; // the active EventSource, closed on navigation
// Live heartbeat for a run that is active but silent (thinking / resuming): a
// 1s ticker updates "#hb-line" with a spinner + elapsed so it reads as alive,
// not hung. hbState is refreshed by each paintRun; hbTimer runs while on a run.
let hbState = { active: false, role: 'worker', last: 0, spin: 0 };
// A run is live per the dir-aware `live` flag the server stamps; the fold's
// `finished` stays false for a run whose worker was killed, so using it alone
// painted a ticking "working…" heartbeat under a "stale" header.
function notLive(s) { return typeof s.live === 'boolean' ? !s.live : !!s.finished; }
let hbTimer = null;
let hubTimer = null; // the hub's list refresh, cleared by closeLive()
let hubVisWake = null; // the hub's visibilitychange repaint, cleared with it
const HUB_POLL_MS = 4000;
const HB_FRAMES = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏';
function hbTick() {
  const line = document.getElementById('hb-line');
  if (!line || !hbState.active) return;
  const secs = Math.floor((Date.now() - hbState.last) / 1000);
  const glyph = HB_FRAMES[hbState.spin % HB_FRAMES.length];
  line.textContent = `${glyph} ${hbState.role} working… ${secs}s`;
}
let activeOverlayClose = null; // an open modal dialog's dismisser, closed on navigation

// --- theme -------------------------------------------------------------------
if (localStorage.getItem('a6-theme') === 'light') document.documentElement.classList.add('light');
function toggleTheme() {
  const on = document.documentElement.classList.toggle('light');
  localStorage.setItem('a6-theme', on ? 'light' : 'dark');
}

// --- nav rail collapse ---------------------------------------------------------
if (localStorage.getItem('a6-rail') === 'min') document.documentElement.classList.add('rail-min');
function railArrow() {
  const a = document.getElementById('rail-arrow');
  if (a) a.textContent = document.documentElement.classList.contains('rail-min') ? '»' : '«';
}
function toggleRail() {
  const on = document.documentElement.classList.toggle('rail-min');
  localStorage.setItem('a6-rail', on ? 'min' : '');
  railArrow();
}
railArrow(); // reflect the persisted state on load

// --- PWA + notifications -----------------------------------------------------
// Install the service worker so the page is an installable PWA (manifest + SW).
// No Web Push / VAPID: OS notifications are the foreground Notification API only.
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => { navigator.serviceWorker.register('/sw.js').catch(()=>{}); });
}
// Ask for OS-notification permission on a user gesture (browsers block passive
// requests). A granted permission lets machine.notify/end pop a desktop/PWA
// notification even when the tab is backgrounded (on desktop).
function enableNotifications() {
  if (!('Notification' in window)) { toast('notifications not supported', true); return; }
  Notification.requestPermission().then(p => toast(p === 'granted' ? 'notifications on' : 'notifications ' + p));
}
// Fire an OS notification when permitted; always safe (never throws into a repaint).
function osNotify(title, body) {
  try { if ('Notification' in window && Notification.permission === 'granted') new Notification(title, { body: body || '', icon: '/icon.svg' }); } catch (_) {}
}

// --- helpers -----------------------------------------------------------------
const el = (t, cls, txt) => { const e = document.createElement(t); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const esc = s => (s == null ? '' : String(s));
async function getJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error((await r.json().catch(()=>({error:r.statusText}))).error || r.statusText); return r.json(); }
async function postJSON(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok || data.ok === false) throw new Error(data.error || r.statusText);
  return data;
}
function toast(msg, bad) { const t = el('div', 'toast' + (bad ? ' bad' : ''), msg); document.body.appendChild(t); setTimeout(() => t.remove(), 4000); }
// Mirrors viewmodel/format.py format_cost precision (cents >= $1, else 4dp); keep in sync.
// partial: the figure is a known lower bound (unpriced spend) -> '~' prefix,
// and ~$0.0000 is information where a clean $0 stays terse (format_cost's rule).
function fmtUsd(u, partial) {
  // Mirrors Python format_cost exactly (no terse "$0" special-case): ~ = a
  // partial lower bound, 2 decimals at/above ~$1, else 4. A genuinely clean $0
  // is BLANKED by the hub callers (like the CLI/TUI hubs), not shown here.
  const p = partial ? '~' : '';
  return (u || 0) >= 0.995 ? p + '$' + Number(u || 0).toFixed(2) : p + '$' + Number(u || 0).toFixed(4);
}
function when(ts) { if (!ts) return ''; const d = new Date(ts * 1000); return d.toLocaleString(); }
function setCrumb(t) { crumb.textContent = t || ''; }
function closeLive() {
  if (live) { live.close(); live = null; }
  if (hubTimer) { clearInterval(hubTimer); hubTimer = null; }
  if (hubVisWake) { document.removeEventListener('visibilitychange', hubVisWake); hubVisWake = null; }
  if (hbTimer) { clearInterval(hbTimer); hbTimer = null; }
  hbState.active = false;
}
function closeOverlay() { if (activeOverlayClose) activeOverlayClose(); }
function pill(level, label) { return el('span', 'pill ' + esc(level || 'neutral'), esc(label)); }

// One owner for anything clickable that is not a native control: a keyboard
// user gets the same activation (Tab to focus, Enter/Space to fire) and a
// screen reader gets a role + name. Native <a>/<button>/<input> never need it.
function actionable(elem, activate, label) {
  elem.setAttribute('role', 'button');
  elem.tabIndex = 0;
  if (label) elem.setAttribute('aria-label', label);
  elem.onclick = activate;
  elem.onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); }
  };
}

function setTab(name) {
  document.querySelectorAll('nav.tabs a, aside.rail .rail-nav a').forEach(a => a.classList.toggle('active', a.dataset.tab === name));
}

// --- router ------------------------------------------------------------------
let booted = false; // the one-shot deep-link to `agent6 web <target>` ran
// Navigation generation: route() is an async, directly re-entrant hashchange
// handler, and every render helper awaits a fetch BEFORE its first DOM write /
// EventSource assignment. A superseded render's continuation must bail at each
// await boundary or it paints the WRONG view over the current one, appends a
// duplicate wmenu button, and overwrites `live` -- orphaning the current
// view's stream (whose stale onmessage could later closeLive() the visible
// view when the old run finishes).
let routeGen = 0;
async function route() {
  const gen = ++routeGen;
  closeLive();
  closeOverlay();
  document.querySelectorAll('.wmenu-btn').forEach(b => b.remove()); // header-mounted by the run view
  const h = location.hash.replace(/^#/, '') || '/';
  // First load with no hash: honor the CLI's target (`agent6 web <run-id>`
  // opens that run; a machine name its machine). Explicit hashes win.
  if (!booted) {
    booted = true;
    if (h === '/') {
      try {
        const meta = await getJSON('/api/meta');
        if (gen !== routeGen) return; // superseded while fetching
        if (meta.target && meta.target_kind) {
          location.hash = '#/' + meta.target_kind + '/' + encodeURIComponent(meta.target);
          return; // the hashchange re-enters route()
        }
      } catch (_) { /* no meta: fall through to the hub */ }
    }
  }
  const parts = h.split('/').filter(Boolean); // e.g. ['session','abc']
  try {
    if (parts.length === 0) { setTab('hub'); await renderHub(undefined, gen); }
    else if (parts[0] === 'machines') { setTab('machines'); await renderHub('machines', gen); }
    else if (parts[0] === 'config') { setTab('config'); await renderConfig(gen); }
    else if (parts[0] === 'session' && parts[1]) { setTab('hub'); await renderRun(decodeURIComponent(parts[1]), undefined, gen); }
    else if (parts[0] === 'conversation' && parts[1]) { setTab('hub'); await renderConversation(decodeURIComponent(parts[1]), gen); }
    else if (parts[0] === 'machine' && parts[1]) { setTab('machines'); await renderMachine(decodeURIComponent(parts[1]), gen); }
    else if (parts[0] === 'draft' && parts[1]) { const n = decodeURIComponent(parts[1]); setTab('machines'); await renderRun(n, { base: '/api/draft/' + encodeURIComponent(n), readOnly: true, title: 'Machine draft', crumb: 'draft ' + n }, gen); }
    else { view.innerHTML = ''; view.appendChild(el('div', 'empty', 'not found')); }
  } catch (e) {
    if (gen !== routeGen) return; // a superseded render's late error must not paint
    view.innerHTML = '';
    view.appendChild(el('div', 'empty err', 'error: ' + e.message));
  }
}
window.addEventListener('hashchange', route);

// --- hub ---------------------------------------------------------------------
// /parallel model-id autocomplete for the new-work composer. When the task text
// starts with `/parallel ` and the caret sits in the spec token, offer the known
// model ids (GET /api/config/suggest/parallel.models — exactly the set
// run --parallel accepts) filtered by the comma-fragment under the caret; click,
// Enter, or ↑/↓+Enter inserts it. The web analogue of the config editor's
// datalist (a <textarea> can't carry a native datalist).
function attachParallelSuggest(task, root) {
  let models = null;           // null=unfetched, []=in-flight/empty, else the list
  let box = null, items = [], active = -1;
  const ensureModels = () => {
    if (models !== null) return;
    models = [];                // sentinel: fetch at most once per composer
    getJSON('/api/config/suggest/parallel.models').then(d => { models = d.values || []; render(); }).catch(() => {});
  };
  const frag = () => {          // the comma-fragment under the caret, or null
    const v = task.value, caret = task.selectionStart;
    const m = /^\/parallel\s+/.exec(v);
    if (!m) return null;
    const start = m[0].length;
    let end = start;
    while (end < v.length && !/\s/.test(v[end])) end++;
    if (caret < start || caret > end) return null;   // caret outside the spec token
    const fragStart = start + v.slice(start, caret).lastIndexOf(',') + 1;
    return { fragStart, fragEnd: caret, text: v.slice(fragStart, caret) };
  };
  const close = () => { if (box) { box.remove(); box = null; } items = []; active = -1; };
  const insert = (model) => {
    const f = frag(); if (!f) { close(); return; }
    const v = task.value;
    task.value = v.slice(0, f.fragStart) + model + v.slice(f.fragEnd);
    const pos = f.fragStart + model.length;
    task.setSelectionRange(pos, pos); close(); task.focus();
  };
  const render = () => {
    const f = frag();
    if (!f) { close(); return; }
    ensureModels();
    const q = f.text.toLowerCase();
    const hit = m => m.toLowerCase();
    items = models.filter(m => hit(m).startsWith(q)).concat(models.filter(m => hit(m).includes(q) && !hit(m).startsWith(q))).slice(0, 8);
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) { box = el('div', 'ac-pop'); root.appendChild(box); }
    box.textContent = '';
    items.forEach((m, i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''), m);
      o.onmousedown = (e) => { e.preventDefault(); insert(m); };
      box.appendChild(o);
    });
  };
  task.addEventListener('input', () => { active = -1; render(); });
  task.addEventListener('click', render);
  task.addEventListener('blur', () => setTimeout(close, 120));
  // Returns true when the popup consumed the key (caller must not also act on it).
  return { onKeyDown(e) {
    if (!box || !items.length) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

// The new-work composer, docked at the bottom of the Sessions page: task text +
// mode + preset + Start (Enter starts, Shift+Enter newline). `presets` is the
// hub payload's list; the first option keeps the config's own preset.
function newWorkDock(presets) {
  const root = el('div', 'composer dock dock-fixed');
  const row = el('div', 'row');
  const task = el('textarea', 'field'); task.placeholder = 'task / question…';
  const mode = el('select', 'field'); mode.style.flex = '0 0 auto'; mode.style.width = 'auto';
  for (const m of ['run', 'plan', 'ask']) { const o = el('option', null, m); o.value = m; mode.appendChild(o); }
  const preset = el('select', 'field'); preset.style.flex = '0 0 auto'; preset.style.width = 'auto';
  preset.title = 'config preset for this run (a preset cannot change mid-run)';
  const dflt = el('option', null, 'preset: config default'); dflt.value = ''; preset.appendChild(dflt);
  for (const p of (presets || [])) { const o = el('option', null, p); o.value = p; preset.appendChild(o); }
  const go = el('button', 'primary', 'Start');
  const start = async () => {
    if (!task.value.trim()) return;
    go.disabled = true;
    try {
      const d = await postJSON('/api/new', { mode: mode.value, task: task.value, preset: preset.value });
      if (d.session_id) location.hash = '#/session/' + encodeURIComponent(d.session_id);
    } catch (e) { toast(e.message, true); go.disabled = false; }
  };
  go.onclick = start;
  const ac = attachParallelSuggest(task, root);
  task.onkeydown = (e) => {
    if (ac.onKeyDown(e)) return;   // the /parallel suggestion popup took the key
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); start(); }
  };
  row.appendChild(task); row.appendChild(mode); row.appendChild(preset); row.appendChild(go);
  root.appendChild(growGrip(task));
  root.appendChild(row);
  root.appendChild(el('div', 'hint', 'Enter starts the run / plan / ask · Shift+Enter newline · '
    + '/parallel [N|models] <task> fans out lanes (repeat to queue more)'));
  return root;
}

// The create-machine composer, docked at the bottom of the Machines page.
function createMachineDock() {
  const root = el('div', 'composer dock dock-fixed');
  const row = el('div', 'row');
  const ct = el('textarea', 'field'); ct.placeholder = 'describe a machine to create…';
  const cbtn = el('button', 'primary', 'Create machine');
  const create = async () => {
    if (!ct.value.trim()) return; cbtn.disabled = true;
    try { const d = await postJSON('/api/machine/create', { task: ct.value }); ct.value=''; if (d.draft) location.hash = '#/draft/' + encodeURIComponent(d.draft); }
    catch (e) { toast(e.message, true); cbtn.disabled = false; }
  };
  cbtn.onclick = create;
  ct.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); create(); } };
  row.appendChild(ct); row.appendChild(cbtn);
  root.appendChild(growGrip(ct));
  root.appendChild(row);
  root.appendChild(el('div', 'hint', 'Enter creates a machine draft from the description · Shift+Enter newline'));
  return root;
}

// A list card: h2 title + one clickable row per entry.
function listCard(title, entries, empty, paint) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, title));
  const list = el('div', 'list');
  if (!entries.length) list.appendChild(el('div', 'empty', empty));
  for (const e of entries) {
    const it = el('div', 'item');
    const g = el('div', 'grow');
    it.appendChild(g);
    paint(e, it, g);
    if (it.onclick) actionable(it, it.onclick,
      [...g.children].map(c => c.textContent).join(' · ').slice(0, 120));
    list.appendChild(it);
  }
  card.appendChild(list);
  return card;
}

function sessionsCard(sessions) {
  const card = listCard('Sessions', sessions, 'no sessions yet', (r, it, g) => {
    it.onclick = () => location.hash = '#/session/' + encodeURIComponent(r.id);
    g.appendChild(el('div', 'title', (r.winner ? '★ ' : '') + (r.task || '(no task)')));
    // A genuinely clean $0 (no spend, not partial) is blanked, like the CLI/TUI
    // hub rows; an all-unpriced ~$0 still shows (spend happened, price unknown).
    const cost = (!r.usd && !r.usd_partial) ? '' : ' · ' + fmtUsd(r.usd, r.usd_partial);
    g.appendChild(el('div', 'sub', `${esc(r.mode)} · ${esc(r.id)} · ${when(r.mtime)}${cost}`));
    it.appendChild(pill(r.level, r.label || r.status)); // the server's one shared label + level
  });
  const prune = el('button', 'danger'); prune.textContent = 'Prune merged runs'; prune.style.marginTop = '10px';
  prune.onclick = async () => { try { const d = await postJSON('/api/sessions/prune', {}); toast(d.message || 'pruned'); route(); } catch (e) { toast(e.message, true); } };
  card.appendChild(prune);
  const rmAsks = el('button', 'danger'); rmAsks.textContent = 'Clear saved asks';
  rmAsks.style.marginTop = '10px'; rmAsks.style.marginLeft = '6px';
  rmAsks.onclick = async () => {
    if (!confirm('Delete every saved ask?')) return;
    try { const d = await postJSON('/api/sessions/rm_asks', {}); toast(d.message || 'cleared'); route(); }
    catch (e) { toast(e.message, true); }
  };
  card.appendChild(rmAsks);
  return card;
}

function machinesCard(machines) {
  return listCard('Machines', machines, 'no machine instances', (m, it, g) => {
    it.onclick = () => location.hash = '#/machine/' + encodeURIComponent(m.name);
    g.appendChild(el('div', 'title', m.machine || m.name));
    g.appendChild(el('div', 'sub', `${m.name} · at ${esc(m.current || '?')} · ${when(m.mtime)}`));
    it.appendChild(pill(m.level, m.label || m.status)); // keep the reason (failed · why)
  });
}

function draftsCard(drafts) {
  return listCard('Machine drafts', drafts, '', (d, it, g) => {
    it.onclick = () => location.hash = '#/draft/' + encodeURIComponent(d.id);
    g.appendChild(el('div', 'title', d.task || d.id));
    g.appendChild(el('div', 'sub', `draft · ${esc(d.id)} · ${when(d.mtime)}`));
    it.appendChild(pill(d.level, d.label || d.status)); // keep the reason (failed · provider_error)
  });
}

function machineFilesCard(files) {
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Run a machine'));
  const frow = el('div', 'form-row');
  for (const mf of files) {
    const b = el('button', null, '▶ ' + mf.name);
    b.onclick = async () => {
      // A machine run spends against your provider from this one click, so
      // confirm the cost first (the deliberate-cost bar the composer's typed
      // Start and the stop dialog already hold).
      if (!confirm('Run ' + mf.name + '? It starts a paid machine run now and spends against your provider.')) return;
      try { await postJSON('/api/machine/run', { file: mf.path }); toast('started ' + mf.name); setTimeout(route, 800); } catch (e) { toast(e.message, true); }
    };
    frow.appendChild(b);
  }
  card.appendChild(frow);
  return card;
}

async function renderHub(focus, gen) {
  setCrumb('');
  const data = await getJSON('/api/hub');
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const machinesTab = focus === 'machines';
  // Full-width listing stack; the tab's composer docks at the bottom of the
  // viewport (new work on Sessions, create-machine on Machines).
  const build = (d) => {
    const lists = el('div', 'grid');
    if (machinesTab) {
      if ((d.machine_files || []).length) lists.appendChild(machineFilesCard(d.machine_files));
      lists.appendChild(machinesCard(d.machines));
      if ((d.drafts || []).length) lists.appendChild(draftsCard(d.drafts));
    } else {
      lists.appendChild(sessionsCard(d.sessions));
    }
    return lists;
  };
  let lists = build(data);
  view.appendChild(lists);
  view.appendChild(machinesTab ? createMachineDock() : newWorkDock(data.presets));
  // The hub painted once and never again, so a lane that finished, failed, or
  // crashed kept its "running" pill until a manual reload -- and clicking the
  // already-active tab does not re-enter route(). Refresh the LISTS only: a
  // whole-view repaint would discard text typed into the dock's composer.
  const refreshLists = async () => {
    if (document.hidden) return; // a background tab has nobody to mislead
    if (gen !== undefined && gen !== routeGen) return; // superseded; closeLive() clears us
    try {
      const next = await getJSON('/api/hub');
      if (gen !== undefined && gen !== routeGen) return;
      const fresh = build(next);
      lists.replaceWith(fresh);
      lists = fresh;
    } catch (_) { /* transient: keep the last good paint */ }
  };
  hubTimer = setInterval(refreshLists, HUB_POLL_MS);
  // Re-focusing the tab otherwise showed the pre-blur pills (the interval
  // skips hidden ticks) for up to a full poll period.
  hubVisWake = () => { refreshLists(); };
  document.addEventListener('visibilitychange', hubVisWake);
}

// --- conversation ------------------------------------------------------------
// Renders a /conversation payload: folded transcript items whose lines are
// [text, style] spans from the shared renderer (viewmodel.transcript_style), so
// the web shows exactly what the CLI stream and the TUI conversation view show.
// The detail level cycles collapsed -> expanded -> hidden (persisted); an item
// with a longer form (clipped tool output, folded thinking) expands on click.
const DETAIL_CYCLE = { collapsed: 'expanded', expanded: 'hidden', hidden: 'collapsed' };
function tailStr(s, n) { return s.length <= n ? s : '…' + s.slice(-n); }
function firstLine(s, n) { const t = String(s == null ? '' : s).split('\n')[0]; return t.length > n ? t.slice(0, n - 1) + '…' : t; }
// `box` is the scroll container, `body` the host the items render into.
function makeConv(url, box, body) {
  const conv = {
    items: [], open: new Set(),
    detail: localStorage.getItem('a6-detail') || 'collapsed',
    timer: null,
    finished: false, // set by setLive: an ended run/machine gets past-tense empty text
  };
  const itemsHost = el('div', 'conv');
  const liveHost = el('div', 'conv conv-live');
  liveHost.style.display = 'none';
  body.appendChild(itemsHost); body.appendChild(liveHost);
  const following = () => box.scrollTop + box.clientHeight >= box.scrollHeight - 40;
  // A run/machine that produced no conversation: "yet ... as it streams" while it
  // can still stream, past tense once it has ended (a terminal tool-only machine).
  const emptyNote = () => conv.finished
    ? 'this run made no conversation'
    : 'no conversation yet; it appears as the run streams';

  const paintItems = () => {
    const follow = following();
    itemsHost.innerHTML = '';
    let shown = 0;
    conv.items.forEach((it, i) => {
      if ((it.kind === 'thinking' || it.kind === 'tool') && conv.detail === 'hidden' && !conv.open.has(i)) return;
      const expanded = conv.detail === 'expanded' || conv.open.has(i);
      const lines = expanded && it.full ? it.full : it.lines;
      const div = el('div', 'ci' + (it.full ? ' exp' : ''));
      if (it.full) {
        div.title = expanded ? 'click to collapse' : 'click to expand';
        div.onclick = () => { if (conv.open.has(i)) conv.open.delete(i); else conv.open.add(i); paintItems(); };
      }
      for (const line of lines) {
        const ln = el('div');
        for (const [text, style] of line) ln.appendChild(el('span', 's-' + style, text));
        if (!line.length) ln.appendChild(document.createTextNode(' '));
        div.appendChild(ln);
      }
      itemsHost.appendChild(div); shown++;
    });
    // The empty note yields to the live pane: streamed text under a
    // "no conversation yet" banner reads as a contradiction.
    if (!shown && liveHost.style.display === 'none') {
      itemsHost.appendChild(el('div', 'muted conv-empty', emptyNote()));
    }
    if (follow) box.scrollTop = box.scrollHeight;
  };

  conv.refresh = async () => {
    if (!box.isConnected) return; // navigated away: don't fetch or paint stale
    let data; try { data = await getJSON(url); } catch (_) { return; }
    if (!box.isConnected) return;
    conv.items = data.items || [];
    paintItems();
  };
  conv.poke = () => { // debounced re-fold on an SSE change signal
    if (conv.timer) return;
    conv.timer = setTimeout(() => { conv.timer = null; conv.refresh(); }, 900);
  };
  // The in-progress turn under the folded items (streamed thinking/text from
  // the SessionState SSE frame): the analogue of the TUI's docked live pane. The
  // live "thinking…" marker always shows; the reasoning text itself streams
  // only at the expanded detail level (same rule as the TUI).
  conv.setLive = (s) => {
    conv.finished = notLive(s); // steers emptyNote()'s tense (may run before paintItems)
    const r = s.last_role;
    const follow = following();
    liveHost.innerHTML = '';
    const note = itemsHost.querySelector('.conv-empty');
    if (notLive(s)) {
      liveHost.style.display = 'none';
      if (note) { note.textContent = emptyNote(); note.style.display = ''; } // re-tense if already painted
      return;
    }
    if (s.status === 'waiting') {
      // Blocked on the operator (the prompt card above): no thinking, no tool
      // running, so neither pulse may say so -- nor "appears as the run
      // streams" over a run that has not started its first turn.
      liveHost.style.display = '';
      if (note) note.style.display = 'none';
      liveHost.appendChild(el('div', 'muted', '· waiting for your answer'));
      return;
    }
    if (!r) {
      liveHost.style.display = 'none';
      if (note) { note.textContent = emptyNote(); note.style.display = ''; }
      return;
    }
    const think = r.streamed_thinking, text = r.streamed_text;
    liveHost.style.display = '';
    if (note) note.style.display = 'none'; // the live pane replaces the empty note
    if (think || text) {
      if (think) {
        const line = el('div');
        line.appendChild(el('span', 'lt', '· thinking… '));
        if (conv.detail === 'expanded') line.appendChild(el('span', 's-thinking', tailStr(think, 1600)));
        liveHost.appendChild(line);
      }
      if (text) liveHost.appendChild(el('div', null, tailStr(text, 1600)));
    } else {
      const hb = el('div', 'muted'); hb.id = 'hb-line'; liveHost.appendChild(hb); hbTick();
    }
    if (follow) box.scrollTop = box.scrollHeight;
  };
  conv.detailButton = () => {
    const b = el('button', 'mini', 'detail: ' + conv.detail);
    b.onclick = () => {
      conv.detail = DETAIL_CYCLE[conv.detail];
      localStorage.setItem('a6-detail', conv.detail);
      b.textContent = 'detail: ' + conv.detail;
      conv.open.clear();
      paintItems();
    };
    return b;
  };
  return conv;
}

// A titled conversation card, the detail toggle in its (non-scrolling) header
// and the items scrolling in .conv-box below it; used by the run view (main
// pane), the full-page view, and the machine view's current-state pane.
function convCard(url, title, cls) {
  const card = el('div', 'card conv-card ' + (cls || ''));
  const hrow = el('div', 'card-head-row');
  hrow.appendChild(el('h2', null, title));
  const box = el('div', 'conv-box');
  const body = el('div');
  box.appendChild(body);
  const conv = makeConv(url, box, body);
  hrow.appendChild(conv.detailButton());
  card.appendChild(hrow); card.appendChild(box);
  return { card, conv, box };
}

// A horizontal grip along a docked composer's top edge: dragging it up grows
// the text entry (the native grip sits bottom-right, where the screen ends).
function growGrip(ta) {
  const g = el('div', 'grow-grip');
  g.title = 'drag to resize';
  g.onpointerdown = (e) => {
    e.preventDefault();
    g.setPointerCapture(e.pointerId);
    g.classList.add('dragging');
    const startY = e.clientY;
    const startH = ta.getBoundingClientRect().height;
    g.onpointermove = (ev) => {
      ta.style.height = Math.round(Math.max(46, Math.min(window.innerHeight * 0.5, startH + startY - ev.clientY))) + 'px';
    };
    g.onpointerup = (ev) => {
      g.releasePointerCapture(ev.pointerId);
      g.classList.remove('dragging');
      g.onpointermove = null; g.onpointerup = null;
    };
  };
  return g;
}

// The text /undo takes back, consumed by the next session view's composer.
let pendingComposerFill = '';

// Fill a composer through an edit the browser records, so the native undo
// stack (Ctrl-Z / Cmd-Z) survives programmatic fills like history recall and
// slash completion. execCommand is deprecated but remains the only widely
// implemented way to write a textarea's value AS a user edit; the fallback
// keeps the fill working, minus undo.
function fillAsEdit(ta, text) {
  ta.focus();
  ta.setSelectionRange(0, ta.value.length);
  let ok = false;
  try { ok = document.execCommand('insertText', false, text); } catch (_) { ok = false; }
  if (!ok) {
    ta.value = text;
    ta.dispatchEvent(new Event('input', { bubbles: true }));
  }
  ta.setSelectionRange(ta.value.length, ta.value.length);
}

// A lightweight lexical highlighter for an approval's command: quoted strings,
// flags, and shell operators get classes; everything else is escaped text.
function highlightCmd(text) {
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const re = /("(?:[^"\\]|\\.)*"|'[^']*')|(\s--?[A-Za-z][\w-]*)|(\|\||&&|\||;|>>|2>|>|<)/g;
  let out = '', last = 0, m;
  while ((m = re.exec(text)) !== null) {
    out += esc(text.slice(last, m.index));
    if (m[1]) out += '<span class="tok-str">' + esc(m[1]) + '</span>';
    else if (m[2]) out += '<span class="tok-flag">' + esc(m[2]) + '</span>';
    else out += '<span class="tok-op">' + esc(m[3]) + '</span>';
    last = m.index + m[0].length;
  }
  return out + esc(text.slice(last));
}

// The steer directives a session composer can complete, with one-line help:
// a verbatim mirror of agent6.directive.STEER_COMMANDS (drift-pinned by
// tests/web/test_steer_completion.py). /compact only acts on a live session,
// so the resume composer offers /pin and /parallel alone.
const STEER_COMMANDS = [
  ['/pin', 'pin an instruction that survives compaction: /pin <text>'],
  ['/compact', 'compact the context now; /compact <focus> steers the summary'],
  ['/parallel', 'fan out lanes: /parallel [N|models] <task> (repeat to queue more)'],
  ['/restate', 'restate the conversation since your last message (local, no model call)'],
  ['/undo', 'fork back to before your last message (the text returns to edit and resend)'],
  ['/btw', 'ask a question beside the run: /btw <question> (answers inline, later)'],
  ['/shells', 'background commands this run started, and how they ended'],
];
// Slash-command completion for a session composer: while the FIRST word is
// being typed (`/…`, no whitespace yet), the matching directives with their
// help. Same popup contract as attachParallelSuggest; Tab (or Enter/click on
// a highlighted row) completes the word plus a trailing space.
function attachCommandSuggest(ta, root, liveNow) {
  let box = null, items = [], active = -1;
  const word = () => {
    const v = ta.value;
    return v.startsWith('/') && !/\s/.test(v) ? v : null;
  };
  const close = () => { if (box) { box.remove(); box = null; } items = []; active = -1; };
  const insert = (cmd) => {
    fillAsEdit(ta, cmd + ' '); // an undoable edit; its input event hands the /parallel popup over
    close();
  };
  const render = () => {
    const w = word();
    if (w === null) { close(); return; }
    items = STEER_COMMANDS.filter(([c]) => (liveNow() || (c !== '/compact' && c !== '/btw')) && c.startsWith(w));
    if (!items.length) { close(); return; }
    if (active >= items.length) active = -1;
    if (!box) { box = el('div', 'ac-pop'); root.appendChild(box); }
    box.textContent = '';
    items.forEach(([c, help], i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''));
      o.appendChild(el('span', null, c));
      o.appendChild(el('span', 'muted', ' · ' + help));
      o.onmousedown = (e) => { e.preventDefault(); insert(c); };
      box.appendChild(o);
    });
  };
  ta.addEventListener('input', () => { active = -1; render(); });
  ta.addEventListener('click', render);
  ta.addEventListener('blur', () => setTimeout(close, 120));
  // Returns true when the popup consumed the key (caller must not also act on it).
  return { onKeyDown(e) {
    if (!box || !items.length) return false;
    if (e.key === 'ArrowDown') { e.preventDefault(); active = (active + 1) % items.length; render(); return true; }
    if (e.key === 'ArrowUp') { e.preventDefault(); active = (active - 1 + items.length) % items.length; render(); return true; }
    if (e.key === 'Tab') { e.preventDefault(); insert(items[active >= 0 ? active : 0][0]); return true; }
    if (e.key === 'Enter' && active >= 0) { e.preventDefault(); insert(items[active][0]); return true; }
    if (e.key === 'Escape') { e.preventDefault(); close(); return true; }
    return false;
  } };
}

// Ctrl-R in a session composer: search this session's past messages (the
// task, then every steer -- journal-read via the conversation payload, so
// resumes and steers typed on other surfaces appear). Newest first, one line
// each, repeats collapsed: the same list the CLI and TUI searches show.
// Picking fills the composer for editing (Enter keeps the highlighted match,
// or the typed text itself when nothing matches); nothing is sent.
function openRestate(text) {
  const back = el('div', 'overlay');
  const box = el('div', 'card'); box.style.width = 'min(720px, 92vw)';
  box.appendChild(el('h2', null, 'since your last message'));
  const pre = el('pre');
  pre.textContent = text;
  pre.style.whiteSpace = 'pre-wrap'; pre.style.maxHeight = '60vh';
  pre.style.overflow = 'auto'; pre.style.margin = '0 0 10px';
  box.appendChild(pre);
  const close = el('button', null, 'Close');
  close.onclick = () => back.remove();
  box.appendChild(close);
  back.onclick = (e) => { if (e.target === back) back.remove(); };
  back.appendChild(box); document.body.appendChild(back);
}

function openHistorySearch(entries, onPick) {
  const back = el('div', 'overlay');
  const box = el('div', 'card'); box.style.width = 'min(560px, 92vw)';
  box.appendChild(el('h2', null, 'search past messages'));
  const field = el('input', 'field'); field.placeholder = 'type to narrow…';
  box.appendChild(field);
  const list = el('div', 'hs-list');
  box.appendChild(list);
  const close = () => { activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey); };
  const pick = (t) => { close(); onPick(t); };
  let items = [], active = 0;
  const render = () => {
    const q = field.value.toLowerCase();
    const all = entries.filter(t => t.toLowerCase().includes(q));
    items = all.slice(0, 8);
    if (active >= items.length) active = Math.max(0, items.length - 1);
    list.textContent = '';
    items.forEach((t, i) => {
      const o = el('div', 'ac-item' + (i === active ? ' on' : ''), t);
      o.onmousedown = (e) => { e.preventDefault(); pick(t); };
      list.appendChild(o);
    });
    if (!items.length) list.appendChild(el('div', 'more-note', '(no match)'));
    else if (all.length > items.length) list.appendChild(el('div', 'more-note', '… ' + (all.length - items.length) + ' more (type to narrow)'));
  };
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); if (items.length) { active = (active + 1) % items.length; render(); } }
    else if (e.key === 'ArrowUp') { e.preventDefault(); if (items.length) { active = (active - 1 + items.length) % items.length; render(); } }
    else if (e.key === 'Enter') { e.preventDefault(); pick(items.length ? items[active] : field.value); }
  }
  activeOverlayClose = close; // navigating away dismisses it
  document.addEventListener('keydown', onKey);
  back.onclick = (e) => { if (e.target === back) close(); };
  field.oninput = () => { active = 0; render(); };
  back.appendChild(box); document.body.appendChild(back);
  field.focus();
  render();
}

// The composer bar under a run's conversation. On a LIVE run Enter sends the
// text as a steer (injected at the run's next safe boundary); on a FINISHED
// run Enter resumes the run with the text as the follow-up instruction (empty
// = plain resume), then waits for the resumed worker to take over and
// re-renders. Shift+Enter inserts a newline. setState(s) keeps the mode in
// sync with each SSE frame. Ctrl-R (composer-focused only, so the browser
// keeps its reload elsewhere) opens the past-message search above.
function makeComposer(id) {
  const root = el('div', 'composer');
  const ta = el('textarea', 'field');
  const hint = el('div', 'hint');
  // The preset the next leg continues under (`agent6 resume --preset`): a
  // preset touches any setting, so it changes only between legs; the picker
  // shows only while the composer resumes, filled once from the same list the
  // config editor offers for the `preset` leaf.
  const presetRow = el('div', 'row');
  presetRow.style.display = 'none';
  presetRow.appendChild(el('span', 'sub muted', 'continue under preset'));
  const preset = el('select', 'field'); preset.style.flex = '0 0 auto'; preset.style.width = 'auto';
  const asRecorded = el('option', null, '(as recorded)'); asRecorded.value = ''; preset.appendChild(asRecorded);
  presetRow.appendChild(preset);
  let presetsFilled = false;
  const fillPresets = () => {
    if (presetsFilled) return;
    presetsFilled = true;
    getJSON('/api/config/suggest/preset').then(d => {
      for (const p of (d.values || [])) { const o = el('option', null, p); o.value = p; preset.appendChild(o); }
    }).catch(() => {});
  };
  let finished = null; // unknown until the first SSE frame
  // A run the AGENT ended has nothing to continue, so resume takes an
  // instruction or is refused; every other ending resumes bare.
  let needsWork = false;
  let busy = false;
  const suggest = attachCommandSuggest(ta, root, () => finished === false);
  const models = attachParallelSuggest(ta, root); // the spec token after `/parallel `
  const apply = () => {
    ta.disabled = busy;
    presetRow.style.display = finished ? '' : 'none';
    if (finished) fillPresets();
    if (busy) { hint.textContent = 'resuming…'; return; }
    if (finished && needsWork) {
      ta.placeholder = 'what should it do next…';
      hint.textContent = 'This session finished: Enter resumes it with your instruction · Shift+Enter newline · Ctrl-R past messages';
    } else if (finished) {
      ta.placeholder = 'continue this session…';
      hint.textContent = 'Enter resumes this session with the instruction (empty = just resume) · Shift+Enter newline · Ctrl-R past messages';
    } else {
      ta.placeholder = 'steer this session… (/pin pins an instruction, /compact [focus] compacts)';
      hint.textContent = 'Enter sends the instruction at the session’s next safe boundary · Shift+Enter newline · Ctrl-R past messages';
    }
  };
  const resume = async (text) => {
    busy = true; apply();
    try {
      await postJSON('/api/session/' + encodeURIComponent(id) + '/resume', { text, preset: preset.value });
      toast(preset.value ? 'resuming the run under preset ' + preset.value + '…' : 'resuming the run…');
      // The resume is a detached spawn: wait for it to come LIVE, then re-open
      // the view so the SSE stream and controls come back. Waiting on
      // `finished === false` declared takeover on the first poll, because the
      // parked and stale runs this composer offers resume for are already
      // unfinished — so a resume that died on spawn (its stderr goes to
      // DEVNULL) reported success and the operator saw nothing.
      for (let i = 0; i < 25; i++) {
        await new Promise(r => setTimeout(r, 1000));
        if (!root.isConnected) return; // navigated away
        let s; try { s = await getJSON('/api/session/' + encodeURIComponent(id)); } catch (_) { continue; }
        if (s && s.live === true) { ta.value = ''; route(); return; }
      }
      toast('the resume has not started yet; check `agent6 sessions`', true);
    } catch (e) { toast(e.message, true); }
    busy = false; apply();
  };
  if (pendingComposerFill) {
    const fill = pendingComposerFill;
    pendingComposerFill = '';
    setTimeout(() => fillAsEdit(ta, fill), 0); // after the view attaches
  }
  ta.onkeydown = (e) => {
    if (suggest.onKeyDown(e) || models.onKeyDown(e)) return;
    if (e.key === 'r' && e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
      e.preventDefault(); // Ctrl-R here is history search, not a page reload
      if (busy) return;
      getJSON('/api/session/' + encodeURIComponent(id) + '/conversation').then(d => {
        const seen = new Set(), entries = [];
        for (const t of (d.operator_inputs || []).slice().reverse()) {
          const line = t.split(/\s+/).join(' ').trim();
          if (line && !seen.has(line)) { seen.add(line); entries.push(line); }
        }
        if (!entries.length) { toast('no past messages this session yet', true); return; }
        openHistorySearch(entries, (text) => fillAsEdit(ta, text));
      }).catch(err => toast(err.message, true));
      return;
    }
    if (e.key !== 'Enter' || e.shiftKey) return;
    e.preventDefault();
    if (finished === null || busy) return;
    const text = ta.value.trim();
    if (text === '/restate') {
      // Local and free: rendered from the journal, nothing reaches the model.
      getJSON('/api/session/' + encodeURIComponent(id) + '/restate')
        .then(d => { openRestate(d.text || ''); ta.value = ''; })
        .catch(err => toast(err.message, true));
      return;
    }
    if (text === '/shells') {
      // The view is the Background shells card: bring it up (the phone menu
      // picks it; the desktop page opens the drawer and scrolls to it), and
      // say so when there is nothing to show, as the CLI and TUI do.
      ta.value = '';
      const card = document.querySelector('[data-w="shells"]');
      if (!card || card.style.display === 'none') { toast('no background commands this run'); return; }
      const drawer = document.querySelector('.drawer');
      if (drawer && drawer.classList.contains('closed')) {
        drawer.classList.remove('closed');
        document.querySelector('.details-btn')?.classList.add('active');
        localStorage.setItem('a6-drawer', 'open');
      }
      const pick = document.querySelector('.wmenu button[data-w="shells"]');
      if (pick) pick.click();
      card.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
    if (text === '/undo') {
      // Fork at the state before the last message; follow the fork with the
      // undone text back in the composer to edit and resend. A live run does
      // it at its next boundary (the steer channel); the SSE paint follows
      // the session.undone the loop emits. A finished run forks right here.
      if (!finished) {
        postJSON('/api/session/' + encodeURIComponent(id) + '/steer', { text })
          .then(() => { toast('undo requested; applies at the next step'); ta.value = ''; })
          .catch(err => toast(err.message, true));
        return;
      }
      postJSON('/api/session/' + encodeURIComponent(id) + '/undo', {})
        .then(d => {
          toast('undone: forked to ' + d.new_session_id);
          pendingComposerFill = d.undone_text || '';
          location.hash = '#session/' + encodeURIComponent(d.new_session_id);
        })
        .catch(err => toast(err.message, true));
      return;
    }
    if (!finished) {
      if (!text) return;
      // The server decides what the text WAS: `/compact [focus]` is an
      // out-of-band request, not a steer, and it says so.
      postJSON('/api/session/' + encodeURIComponent(id) + '/steer', { text })
        .then(r => { toast((r && r.message) || 'steer sent'); ta.value = ''; })
        .catch(err => toast(err.message, true));
    } else {
      resume(text);
    }
  };
  root.appendChild(growGrip(ta)); root.appendChild(presetRow); root.appendChild(ta); root.appendChild(hint);
  // `live` is the dir-aware truth (a parked or stale run is not live even
  // though the fold says unfinished); fall back to the fold for a payload
  // that predates it.
  root.setState = (s) => {
    if (busy) return;
    needsWork = s.finished === true && s.end_reason === 'finish_session';
    if (typeof s.live === 'boolean') { finished = !s.live; apply(); }
    else { finished = notLive(s); apply(); } // resume-style composer for any non-live run (parked/stale/ended)
  };
  apply();
  return root;
}

