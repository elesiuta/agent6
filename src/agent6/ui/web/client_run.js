// --- run dashboard -----------------------------------------------------------
async function stopRun(base, label) {
  if (!confirm('Stop ' + label + '? It ends now and can be resumed later.')) return;
  try { await postJSON(base + '/steer', { text: 'abort' }); toast('stopping…'); } catch (e) { toast(e.message, true); }
}

// opts: { base, readOnly, title }: a draft (machine-create authoring log) is
// watched read-only against /api/draft/<name>; a run is driveable at /api/session/<id>.
// The snapshot fetch up front is the existence probe (a bad id used to leave a
// hollow dashboard: the conversation fetch swallowed its 404 and the
// EventSource error is silent) and the first paint, so the view never flashes
// empty while waiting for the first SSE frame.
// The details drawer: the run's context widgets combined into one collapsible,
// drag-resizable panel on the left, so the conversation keeps the focus.
function drawerHandle(drawer) {
  const h = el('div', 'drawer-handle');
  h.title = 'drag to resize · double-click to reset';
  h.onpointerdown = (e) => {
    e.preventDefault();
    h.setPointerCapture(e.pointerId);
    h.classList.add('dragging');
    const startX = e.clientX;
    const startW = drawer.getBoundingClientRect().width;
    h.onpointermove = (ev) => {
      const w = Math.round(Math.max(220, Math.min(window.innerWidth * 0.6, startW + ev.clientX - startX)));
      document.documentElement.style.setProperty('--drawer-w', w + 'px');
      localStorage.setItem('a6-drawer-w', w + 'px');
    };
    h.onpointerup = (ev) => {
      h.releasePointerCapture(ev.pointerId);
      h.classList.remove('dragging');
      h.onpointermove = null; h.onpointerup = null;
    };
  };
  h.ondblclick = () => {
    document.documentElement.style.removeProperty('--drawer-w');
    localStorage.removeItem('a6-drawer-w');
  };
  return h;
}
{ const w = localStorage.getItem('a6-drawer-w'); if (w) document.documentElement.style.setProperty('--drawer-w', w); }

async function renderRun(id, opts, gen) {
  opts = opts || {};
  const base = opts.base || ('/api/session/' + encodeURIComponent(id));
  const snap = await getJSON(base); // throws -> route() shows the error
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint or open a stream
  const readOnly = !!opts.readOnly;
  setCrumb(opts.crumb || id);
  view.innerHTML = '';
  // "paged" only acts on phones: one widget shows at a time there, picked by
  // the floating menu; the desktop drawer ignores it.
  const app = el('div', 'run-app paged');
  const prompts = el('div', 'page-pad'); app.appendChild(prompts); // approval/question boxes surface here
  const cards = { _id: id, _prompts: prompts, _readOnly: readOnly };
  const drawer = el('div', 'grid drawer');
  const mk = (key, title, cls, parent) => { const c = el('div', 'card card-' + key + ' ' + (cls||'')); c.dataset.w = key; const h = el('h2', null, title); c.appendChild(h); if (key === 'head') cards._head_title = h; const body = el('div', 'card-body'); c.appendChild(body); cards[key] = body; (parent || drawer).appendChild(c); return body; };

  // Controls at the TOP so Stop stays reachable without scrolling; the Details
  // toggle folds the drawer away (persisted; default open on wide screens).
  const actions = el('div', 'row wrap page-pad'); actions.style.margin = '10px 22px';
  const dBtn = el('button', 'details-btn', 'Details'); // desktop drawer toggle; phones page widgets instead
  const applyDrawer = (open) => {
    drawer.classList.toggle('closed', !open);
    dBtn.classList.toggle('active', open);
    localStorage.setItem('a6-drawer', open ? 'open' : 'closed');
  };
  dBtn.onclick = () => applyDrawer(drawer.classList.contains('closed'));
  actions.appendChild(dBtn);
  if (!readOnly) {
    const post = (verb, okMsg) => async () => {
      try { const d = await postJSON('/api/session/' + encodeURIComponent(id) + '/' + verb, {}); toast(d.message || okMsg); }
      catch (e) { toast(e.message, true); }
    };
    const stopBtn = el('button', 'danger', '■ Stop now');
    stopBtn.onclick = () => stopRun('/api/session/' + encodeURIComponent(id), 'the run');
    const stepBtn = el('button', null, 'Stop after step');
    stepBtn.onclick = post('stop_step', 'stopping after the current step');
    const compactBtn = el('button', null, 'Compact context');
    compactBtn.onclick = post('compact', 'compaction requested');
    const mergeBtn = el('button', null, 'Merge'); // no glyph: U+2443 was tofu in common fonts
    mergeBtn.onclick = post('merge', 'merged');
    cards._merge_btn = mergeBtn; // paintRun gates it on the run actually having a branch
    const rmBtn = el('button', 'danger', 'Delete');
    rmBtn.onclick = async () => {
      // History only, and not undoable, so it asks. The CLI refuses a live run.
      // The branch is named only while it exists (the snapshot's run_branch).
      const kept = cards._run_branch ? ' The branch and its commits are kept.' : '';
      if (!confirm('Delete this run\'s history?' + kept)) return;
      try {
        const d = await postJSON('/api/session/' + encodeURIComponent(id) + '/rm', {});
        toast(d.message || 'removed');
        location.hash = '#/';
      } catch (e) { toast(e.message, true); }
    };
    // Execute a finished plan: spawns `agent6 run --from-plan <id>` detached
    // and opens the run; the plan itself is untouched (the composer keeps
    // revising it). paintRun shows this only on a plan with a plan.md.
    const planBtn = el('button', null, 'Run this plan');
    planBtn.style.display = 'none';
    planBtn.onclick = async () => {
      try {
        const d = await postJSON(base + '/run_plan', {});
        toast('run started: ' + d.run_id);
        location.hash = '#session/' + encodeURIComponent(d.run_id);
      } catch (e) { toast(e.message, true); }
    };
    for (const b of [stopBtn, stepBtn, compactBtn, planBtn, mergeBtn, rmBtn]) actions.appendChild(b);
    cards._live_btns = [stopBtn, stepBtn, compactBtn]; // paintRun disables these once finished
    cards._plan_btn = planBtn;
  }
  app.appendChild(actions);

  // The heading is where the MODE belongs; paintRun fills it in from the
  // snapshot. A fixed word was right one time in three.
  mk('head', opts.title || 'Session', ''); // status/summary leads the drawer
  // A planning run's deliverable (plan.md), shown only when there is one.
  mk('plan', 'plan.md', 'scroll');
  mk('tasks', 'Task graph', 'scroll');
  mk('budget', 'Budget', '');
  mk('tools', 'Tool calls', 'scroll');
  // Background commands the run started and how they ended (hidden until one exists).
  mk('shells', 'Background shells', 'scroll');
  mk('diff', 'Latest commit', 'scroll');
  mk('log', 'Event log', 'scroll');

  const cc = convCard(base + '/conversation', 'Conversation', 'card-conv');
  cards._conv = cc.conv;
  cc.card.dataset.w = 'conv';
  const body = el('div', 'run-body');
  body.appendChild(drawer);
  body.appendChild(drawerHandle(drawer));
  body.appendChild(cc.card);
  app.appendChild(body);
  const saved = localStorage.getItem('a6-drawer');
  applyDrawer(saved ? saved === 'open' : window.innerWidth >= 1024);

  // The phone widget menu: pick which single widget the page shows.
  const entries = [['conv', 'Conversation'], ['head', 'Overview'], ['plan', 'plan.md'],
                   ['tasks', 'Task graph'], ['budget', 'Budget'], ['tools', 'Tool calls'],
                   ['shells', 'Background shells'], ['diff', 'Latest commit'], ['log', 'Event log']];
  const wbtn = el('button', 'wmenu-btn', '☰');
  wbtn.title = 'widgets';
  const wmenu = el('div', 'wmenu'); wmenu.style.display = 'none';
  const setW = (key) => {
    app.querySelectorAll('[data-w]').forEach(c => c.classList.toggle('w-active', c.dataset.w === key));
    wmenu.querySelectorAll('button').forEach(mb => mb.classList.toggle('w-on', mb.dataset.w === key));
  };
  for (const [key, label] of entries) {
    const mb = el('button', null, label); mb.dataset.w = key;
    mb.onclick = () => { setW(key); wmenu.style.display = 'none'; window.scrollTo(0, 0); };
    wmenu.appendChild(mb);
  }
  wbtn.onclick = () => { wmenu.style.display = wmenu.style.display === 'none' ? '' : 'none'; };
  setW('conv');
  // The button lives in the header (next to the theme toggle) so the two share
  // one row and skin; route() removes it since clearing #view won't.
  document.querySelector('header').appendChild(wbtn);
  app.appendChild(wmenu);

  if (!readOnly) {
    // The composer replaces the steer dialog: steer while live, resume when
    // done. Docked at the bottom of the view.
    const composer = makeComposer(id);
    composer.classList.add('dock');
    app.appendChild(composer);
    cards._composer = composer;
  }
  view.appendChild(app);
  paintRun(cards, snap);
  cc.conv.refresh().then(() => {
    // On a phone the page (not the box) scrolls the conversation: open at the tail.
    if (window.innerWidth < 781) window.scrollTo(0, document.body.scrollHeight);
  });

  live = new EventSource(base + '/events');
  // The stream stays open across a finish: a resume from any surface logs into
  // the same file and painting continues (the TUI follows the same way) -- a
  // close-on-finished froze this page on "stopped" while the hub said
  // "running". Only stream_dead (transport: nothing more will come) closes.
  let sawEnd = false;
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    if (s.undone_to) {
      // /undo landed: follow the fork with the undone text back to edit.
      toast('undone: forked to ' + s.undone_to);
      pendingComposerFill = s.undone_text || '';
      location.hash = '#session/' + encodeURIComponent(s.undone_to);
      return;
    }
    paintRun(cards, s);
    hbState.spin++;
    if (s.stream_dead) { closeLive(); setTimeout(() => cc.conv.refresh(), 900); return; }
    if (s.finished && !sawEnd) setTimeout(() => cc.conv.refresh(), 900); // one final fold after last writes flush
    sawEnd = !!s.finished;
  };
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
  live.onerror = () => { /* EventSource auto-retries a live run; leave last paint up */ };
}

// Render the run's unanswered approval / ask_user prompts as actionable boxes.
// Reconcile by id: keep existing boxes so a repaint (any SSE frame) never wipes a
// half-typed free-text answer or drops focus; only add new prompts and remove
// resolved ones.
function paintPrompts(cards, s) {
  const host = cards._prompts;
  // base is the POST prefix: sessions use /api/session/<id>, machines /api/machine/<name>.
  const base = cards._base || ('/api/session/' + encodeURIComponent(cards._id));
  // For a machine, the per-state dir the reasoning (and its prompts) came from.
  // Prompt ids reset per state (approval-1 in every state), so the answer must
  // carry it AND the box key must include it: when the machine advances to a new
  // state, the key changes so the stale box is rebuilt rather than reused with a
  // now-wrong prompt still showing.
  const state = cards._state || '';
  const pfx = state ? state + ':' : '';
  const extra = state ? { state } : {};
  const build = {};
  for (const ap of (s.pending_approvals || [])) {
    if (ap.answered) continue;
    build[pfx + 'ap:' + ap.id] = () => {
      const box = el('div', 'prompt-box');
      const head = ap.head || ap.prompt || 'Approve this action?';
      box.appendChild(el('div', 'q', '? ' + head + (ap.payload ? ':' : '')));
      if (ap.payload) {
        // The command under judgment: fixed width, lexically marked (strings,
        // flags, pipes/redirects); no language detection, just a scanning aid.
        const pre = el('pre', 'cmd');
        pre.innerHTML = highlightCmd(ap.payload);
        box.appendChild(pre);
      }
      const row = el('div', 'form-row');
      const yes = el('button', 'primary', 'Allow');
      const no = el('button', 'danger', 'Deny');
      const send = (answer) => async () => { try { await postJSON(base + '/approve', { id: ap.id, answer, ...extra }); } catch (e) { toast(e.message, true); } };
      yes.onclick = send('yes'); no.onclick = send('no');
      row.appendChild(yes);
      // Only when the prompt says an "allow all" would actually cover its scope:
      // a button that silently answered one call would lie about itself.
      if (ap.standing !== false) { const sess = el('button', 'primary', 'Allow session'); sess.onclick = send('session'); row.appendChild(sess); }
      row.appendChild(no);
      // The fourth answer the run understands: withhold this scope for the rest
      // of the session (the CLI's `x`, the TUI's "Deny all").
      if (ap.standing !== false) { const dall = el('button', 'danger', 'Deny all'); dall.onclick = send('session-deny'); row.appendChild(dall); }
      box.appendChild(row);
      return box;
    };
  }
  for (const q of (s.pending_questions || [])) {
    if (q.answered) continue;
    build[pfx + 'q:' + q.id] = () => {
      // One or more related questions answered together; option buttons FILL that
      // question's field, and a single Submit posts all answers (review first).
      const box = el('div', 'prompt-box');
      // agent6's own start question (the fold says so): name the asker, since the
      // box otherwise reads as the model's.
      if (q.from_harness) box.appendChild(el('div', 'sub muted', 'agent6 asks'));
      const qs = q.questions || [];
      const inputs = [];
      qs.forEach((sub, qi) => {
        const label = (qs.length > 1 ? (qi + 1) + '. ' : '') + (sub.question || 'The agent asked a question');
        box.appendChild(el('div', 'q', label));
        const row = el('div', 'form-row');
        const inp = el('input', 'field'); inp.placeholder = 'pick above or type an answer…'; inp.style.flex = '1';
        for (const opt of (sub.options || [])) {
          const b = el('button', null, opt);
          b.onclick = () => { inp.value = opt; };
          row.appendChild(b);
        }
        row.appendChild(inp); box.appendChild(row);
        inputs.push(inp);
      });
      const send = el('button', 'primary', qs.length > 1 ? 'Submit all' : 'Send');
      send.onclick = async () => {
        const answers = inputs.map(i => i.value.trim());
        // Guard an accidental Send: an all-empty submit would consume this
        // one-shot question and continue the run on fabricated empty input.
        if (answers.every(a => a === '')) { toast('Pick an option or type an answer first.', true); return; }
        try { await postJSON(base + '/answer', { id: q.id, answers, ...extra }); } catch (e) { toast(e.message, true); }
      };
      box.appendChild(send);
      return box;
    };
  }
  const want = new Set(Object.keys(build));
  for (const child of Array.from(host.children)) {
    if (!want.has(child.dataset.key)) child.remove(); // resolved / gone
  }
  const present = new Set(Array.from(host.children).map(c => c.dataset.key));
  for (const key of want) {
    if (present.has(key)) continue; // leave the live box (input + focus) intact
    const box = build[key](); box.dataset.key = key; host.appendChild(box);
  }
}

// Budget and task graph: the live state, or the state as of the step picked
// in the Latest commit card (the server folds the log up to that commit).
function paintDetails(cards, s, asOf) {
  // budget
  const b = s.budget || {};
  cards.budget.innerHTML = '';
  const barRow = (label, frac, text) => {
    const w = el('div'); w.appendChild(el('div', 'sub muted', `${label}: ${text}`));
    const bar = el('div', 'bar' + (frac > 0.85 ? ' warn' : '')); const sp = el('span'); sp.style.width = (frac*100)+'%'; bar.appendChild(sp); w.appendChild(bar); return w;
  };
  // Metered spend vs max_usd (-1 = unlimited); unmetered tokens vs the fallback
  // cap only when that ledger has traffic. The cap re-arms each resume leg, so
  // the bar meters THIS leg's spend (usd_total - usd_prior_legs) while the cost
  // figure stays cumulative -- mirrors ui/tui/app.py render_heartbeat; keep in sync.
  const usdCap = b.usd_cap || 0;
  const legUsd = Math.max(0, (b.usd_total || 0) - (b.usd_prior_legs || 0));
  const usdFrac = usdCap > 0 ? Math.min(1, legUsd / usdCap) : 0;
  const usdText = fmtUsd(b.usd_total, b.usd_partial)
    + (usdCap > 0 ? ((b.usd_prior_legs || 0) > 0 ? ' · leg ' + fmtUsd(legUsd, false) + ' / ' + fmtUsd(usdCap, false) : ' / ' + fmtUsd(usdCap, false))
                  : (usdCap === -1 ? ' (unlimited)' : ''));
  cards.budget.appendChild(barRow('cost', usdFrac, usdText));
  if (b.tokens_unmetered) {
    const fbCap = b.tokens_fallback_cap || 0;
    const fbFrac = fbCap > 0 ? Math.min(1, b.tokens_unmetered / fbCap) : 0;
    const fbText = `${b.tokens_unmetered}${fbCap > 0 ? ' / ' + fbCap : (fbCap === -1 ? ' (unlimited)' : '')} tokens`;
    cards.budget.appendChild(barRow('unmetered', fbFrac, fbText));
  }
  if (b.plan_used_percent > 0) {
    // Subscription plan usage: the account's window fill; the bar meters this
    // run's consumed points against max_percent when one is set, else the
    // account percent itself. Mirrors ui/tui/app.py render_heartbeat.
    const planCap = b.plan_cap || 0;
    const planFrac = planCap > 0 ? Math.min(1, (b.plan_consumed || 0) / planCap)
                                 : Math.min(1, b.plan_used_percent / 100);
    const planText = `${b.plan_used_percent}%`
      + (planCap > 0 ? ` · run ${(b.plan_consumed || 0).toFixed(1)} / ${planCap} pt` : '');
    cards.budget.appendChild(barRow('plan', planFrac, planText));
  }
  // The context-window fill at the last model call (the TUI's `ctx: N%`, the
  // pause menu's readout): the fold's one rule, served as context_pct.
  const ctxPct = typeof s.context_pct === 'number' ? ` · context ${s.context_pct}%` : '';
  cards.budget.appendChild(el('div', 'sub muted', `tokens: in ${b.input_total||0} · out ${b.output_total||0}${ctxPct}`));

  // task tree
  cards.tasks.innerHTML = '';
  const tree = el('div', 'tree');
  if (!(s.tasks||[]).length) tree.appendChild(el('div', 'muted', 'no task graph yet'));
  for (const t of s.tasks || []) {
    const line = el('div', 'node' + (t.is_cursor ? ' cursor' : ''));
    // Mirrors viewmodel/format.py TASK_STATUS_GLYPH (JS can't import it); keep in sync.
    const glyph = { passed:'✓', failed:'✗', in_progress:'▸', pending:'·', skipped:'–', obsolete:'×' }[t.status] || '·';
    line.appendChild(el('span', 'st-' + t.status, '  '.repeat(t.depth) + glyph + ' '));
    line.appendChild(document.createTextNode(t.title));
    tree.appendChild(line);
  }
  cards.tasks.appendChild(tree);
  if (asOf) for (const c of [cards.budget, cards.tasks]) c.prepend(el('div', 'sub muted', `as of iter ${asOf.iteration} · ${String(asOf.sha).slice(0, 7)}`));
}

function paintRun(cards, s) {
  // Stop/compact/answers only mean something on a live run; a finished run
  // ignores the bridge markers. The composer flips to resume mode instead of
  // disabling, and a dead run's prompt boxes reconcile away like the machine
  // view's: the server refuses the POST, so live-looking Allow/Deny beside
  // a "stale" header could only manufacture a red toast.
  const isDead = notLive(s);
  if (!cards._readOnly) paintPrompts(cards, isDead ? {} : s);
  if (cards._live_btns) for (const b of cards._live_btns) b.disabled = isDead;
  // Merge needs a finished run with an unmerged branch: a live run (the server
  // refuses one), an ask / branch_per_run=false run (no branch), or an already-
  // merged branch can't be merged.
  cards._run_branch = s.run_branch || '';
  if (cards._merge_btn) {
    cards._merge_btn.disabled = !notLive(s) || !s.run_branch || !!s.merged_into;
    cards._merge_btn.title = !notLive(s) ? 'the run is still live; stop or let it finish first'
      : s.merged_into ? 'already merged into ' + s.merged_into
      : s.run_branch ? 'merge ' + s.run_branch + (s.base_branch ? ' into ' + s.base_branch : '')
      : 'this run has no branch to merge';
  }
  if (cards._composer) cards._composer.setState(s);
  // header
  // The panel's own heading states the MODE, which is the fact that tells an
  // operator what they are looking at -- and the one the fixed word denied.
  if (cards._head_title && s.mode) cards._head_title.textContent = s.mode;
  cards.head.innerHTML = '';
  const kv = el('div', 'kv');
  const add = (k, v) => { kv.appendChild(el('div', 'k', k)); kv.appendChild(el('div', 'v', v)); };
  add('task', s.user_task || '(none)');
  add('id', s.session_id || cards._id || ''); // older logs carry no session_id in session.start
  add('state', s.status_label || (s.finished ? 'finished' : 'running'));
  // Where the run's work lives and where Merge lands: consecutive spawns chain
  // branches, which is invisible without this line.
  if (s.forked_from) add('forked from', s.forked_from);
  if (s.branch_line) add('branch', s.branch_line);
  if (s.pins && s.pins.length) add('pins', s.pins.join(' | '));
  // What the run is serving: a dev server the agent started is reachable only
  // through `agent6 forward` (the run's network has no way in from outside).
  if (s.ports && s.ports.length) {
    add('serving', s.ports.join(', ') + ' · agent6 forward ' + (s.session_id || cards._id || '') + ' ' + s.ports[0]);
  }
  // Fan-out compare outcome (stamped into a lane's manifest by --parallel's
  // auto-compare): where this lane placed and why. Absent for a non-lane run.
  if (s.compare && typeof s.compare.rank === 'number') {
    const c = s.compare;
    // Mirror format_compare (sessions show / TUI): `rank 1/2 · winner · judge ($0.01)`.
    const parts = ['rank ' + c.rank + '/' + c.of];
    if (c.winner) parts.push('winner');
    if (c.ranked_by) {
      const judged = c.judge_cost_usd > 0 || c.judge_cost_partial;
      parts.push(c.ranked_by + (judged ? ' (' + fmtUsd(c.judge_cost_usd, c.judge_cost_partial) + ')' : ''));
    }
    add('compare', parts.join(' · '));
  }
  cards.head.appendChild(kv);
  // The judge's reason sits under its compare row (as sessions show prints it).
  if (s.compare && s.compare.rationale) {
    cards.head.appendChild(el('div', 'sub muted', 'judge: ' + s.compare.rationale));
  }
  // The same one-line fold the CLI banner and the TUI composer read.
  if (s.policy) cards.head.appendChild(el('div', 'sub muted', esc(s.policy)));
  if (s.last_role) {
    const r = s.last_role;
    cards.head.appendChild(el('div', 'sub muted', `${esc(r.role)} / ${esc(r.model)}${r.in_flight ? ' …' : ''}`));
  }

  const stepState = (cards._diffPick && cards._diffPick.sha && cards._stepState) ? cards._stepState : null;
  paintDetails(cards, stepState || s, stepState ? stepState.as_of : null);

  // conversation: the live in-progress turn paints from this frame at once (a
  // heartbeat ticks via hbTick() on a live-but-silent run so it reads as alive,
  // not hung); completed turns re-fold on a debounce.
  const streaming = s.last_role && (s.last_role.streamed_thinking || s.last_role.streamed_text);
  cards._conv.setLive(s);
  cards._conv.poke();
  hbState = {
    // a "waiting" run is LIVE but blocked on the operator, not working: the
    // conversation shows its own waiting line, so the heartbeat must go quiet.
    active: !notLive(s) && !!s.last_role && !streaming && s.status !== 'waiting',
    role: (s.last_role && s.last_role.role) || 'worker',
    // Server-computed age: replayed history must not read as fresh activity
    // (an arrival anchor showed a 40-minute-wedged run as "working… 3s").
    last: Date.now() - 1000 * (s.last_event_age_s || 0),
    spin: 0,
  };
  hbTick();

  // tools: one clipped line per call (hover shows the full args + result; the
  // conversation carries the whole story), so a long error dump can't flood it.
  cards.tools.innerHTML = '';
  const tbl = el('table', 'tools');
  for (const tc of (s.tool_calls||[]).slice(-30)) {
    const tr = el('tr');
    const d = el('td'); d.appendChild(el('span', 'dot ' + (tc.ok === null ? '' : tc.ok ? 'ok' : 'bad'))); tr.appendChild(d);
    tr.appendChild(el('td', 'name', tc.name));
    const a = el('td', 'args');
    a.textContent = firstLine(tc.args_preview, 90) + (tc.result_summary ? '  → ' + firstLine(tc.result_summary, 90) : '');
    const extra = String(tc.result_summary || '').split('\n').length - 1;
    if (extra > 0) a.appendChild(el('span', 'more-note', ` (+${extra} more line${extra === 1 ? '' : 's'})`));
    a.title = tc.args_preview + (tc.result_summary ? '\n→ ' + tc.result_summary : '');
    tr.appendChild(a);
    tbl.appendChild(tr);
  }
  if (!(s.tool_calls||[]).length) cards.tools.appendChild(el('div', 'muted', 'no tool calls yet'));
  else cards.tools.appendChild(tbl);

  // shells: the roster every surface reads off disk, one line per command
  cards.shells.innerHTML = '';
  const shellsCard = cards.shells.parentElement;
  for (const line of (s.shells||[])) cards.shells.appendChild(el('div', 'shell', line));
  shellsCard.style.display = (s.shells||[]).length ? '' : 'none';

  // log
  cards.log.innerHTML = '';
  const log = el('div', 'log');
  for (const line of (s.log_tail||[]).slice(-200)) log.appendChild(el('div', null, line));
  cards.log.appendChild(log);
  cards.log.scrollTop = cards.log.scrollHeight;

  if (cards._plan_btn) {
    cards._plan_btn.style.display = s.mode === 'plan' && s.plan_md ? '' : 'none';
  }
  // plan.md: the planning run's deliverable (the CLI prints it at the end).
  cards.plan.innerHTML = '';
  const planCard = cards.plan.parentElement;
  if (s.plan_md) {
    const pre = el('pre', 'plan'); pre.textContent = s.plan_md; cards.plan.appendChild(pre);
    planCard.style.display = '';
  } else {
    planCard.style.display = 'none';
  }

  // diff: the latest commit by default; a step selector over the run's
  // chain (newest first) with a cumulative toggle, hidden truthfully when the
  // model owns git (no chain) or nothing is committed yet.
  cards.diff.innerHTML = '';
  const steps = (s.steps || []).slice().reverse();
  if (s.git_control === 'model') {
    cards.diff.appendChild(el('div', 'muted', 'the model owns git in this run: no step chain'));
  } else if (!steps.length) {
    cards.diff.appendChild(el('div', 'muted', 'no commit yet'));
  } else {
    const nav = el('div', 'form-row');
    const sel = document.createElement('select');
    sel.appendChild(new Option('latest commit', ''));
    for (const st of steps) sel.appendChild(new Option('iter ' + st.iteration + ' · ' + st.sha.slice(0, 7) + ' · ' + st.subject, st.sha));
    const cum = document.createElement('input'); cum.type = 'checkbox'; cum.id = 'diff-cumulative';
    const cumLabel = el('label', 'muted', ' cumulative'); cumLabel.htmlFor = 'diff-cumulative';
    const pick = cards._diffPick || { sha: '', cumulative: false };
    sel.value = pick.sha; cum.checked = pick.cumulative;
    nav.appendChild(sel); nav.appendChild(cum); nav.appendChild(cumLabel);
    cards.diff.appendChild(nav);
    const body = el('div');
    cards.diff.appendChild(body);
    const show = async () => {
      cards._diffPick = { sha: sel.value, cumulative: cum.checked };
      body.innerHTML = '';
      if (!sel.value) { cards._stepState = null; paintDetails(cards, s, null); body.appendChild(renderDiff(s.latest_diff || '')); return; }
      try {
        const d = await getJSON('/api/session/' + encodeURIComponent(cards._id) + '/diff?sha=' + encodeURIComponent(sel.value) + '&cumulative=' + (cum.checked ? '1' : '0'));
        body.appendChild(renderDiff(d.patch));
        const st = await getJSON('/api/session/' + encodeURIComponent(cards._id) + '?step=' + encodeURIComponent(sel.value));
        cards._stepState = st; paintDetails(cards, st, st.as_of);
      } catch (e) { body.appendChild(el('div', 'muted', e.message)); }
    };
    sel.onchange = show; cum.onchange = show;
    show();
  }
}

function renderDiff(text) {
  const box = el('pre', 'diff');
  for (const line of text.split('\n')) {
    let cls = null;
    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'del';
    else if (line.startsWith('@@')) cls = 'hunk';
    const span = el('span', cls); span.textContent = line + '\n'; box.appendChild(span);
  }
  return box;
}

// --- conversation page ---------------------------------------------------------
// The run's conversation full-height (the same component the run view embeds),
// live-following: the SessionState /events stream is the change signal; the fold
// re-fetches on it (debounced) and the stream closes once the run finishes.
async function renderConversation(id, gen) {
  const base = '/api/session/' + encodeURIComponent(id);
  const snap = await getJSON(base); // existence probe: throws -> route() shows the error
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  setCrumb('conversation ' + id);
  view.innerHTML = '';
  // The same app shell as the run view, minus the drawer: the conversation
  // fills the view full width and the composer docks at the bottom.
  const app = el('div', 'run-app');
  // This page registers as the run's answering front-end the moment its
  // stream opens, so it must SHOW the prompts it claims to answer: without a
  // host, a run blocked on an approval waited on a page that never painted it.
  const prompts = el('div', 'page-pad'); app.appendChild(prompts);
  const cards = { _id: id, _prompts: prompts };
  const cc = convCard(base + '/conversation', 'Conversation', 'card-conv');
  const body = el('div', 'run-body');
  body.appendChild(cc.card);
  app.appendChild(body);
  const composer = makeComposer(id);
  composer.classList.add('dock');
  app.appendChild(composer);
  view.appendChild(app);
  // Seed from the snapshot BEFORE any stream frame (renderRun's order): the
  // discarded snapshot left a dead run's composer in steer mode (Enter a
  // silent no-op) and the empty note in the future tense -- and a parked or
  // created run gets no SSE frame to ever correct either.
  composer.setState(snap);
  paintPrompts(cards, notLive(snap) ? {} : snap); // the run view's dead-run gating
  cc.conv.setLive(snap);
  await cc.conv.refresh();
  if (gen !== undefined && gen !== routeGen) return; // the refresh reopened the window
  cc.box.scrollTop = cc.box.scrollHeight; // open at the tail, like the TUI
  if (window.innerWidth < 781) window.scrollTo(0, document.body.scrollHeight); // phone: the page scrolls

  live = new EventSource(base + '/events');
  // Same rule as the run view: the stream survives a finish so a resumed leg
  // keeps painting; only stream_dead closes.
  let sawEnd = false;
  live.onmessage = ev => {
    let s; try { s = JSON.parse(ev.data); } catch (_) { return; }
    composer.setState(s);
    paintPrompts(cards, notLive(s) ? {} : s);
    cc.conv.setLive(s);
    cc.conv.poke();
    hbState = {
      // see paintRun: a "waiting" run is live but blocked, not working.
      active: !notLive(s) && !!s.last_role && !(s.last_role.streamed_thinking || s.last_role.streamed_text) && s.status !== 'waiting',
      role: (s.last_role && s.last_role.role) || 'worker',
      last: Date.now() - 1000 * (s.last_event_age_s || 0), // see paintRun: age is server-computed
      spin: hbState.spin + 1,
    };
    hbTick();
    if (s.stream_dead) { closeLive(); setTimeout(() => cc.conv.refresh(), 900); return; }
    if (s.finished && !sawEnd) setTimeout(() => cc.conv.refresh(), 900); // one final fold after last writes flush
    sawEnd = !!s.finished;
  };
  if (!hbTimer) hbTimer = setInterval(() => { hbState.spin++; hbTick(); }, 1000);
}

