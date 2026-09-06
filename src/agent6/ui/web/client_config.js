// --- config ------------------------------------------------------------------
async function renderConfig(gen) {
  setCrumb('config');
  const data = await getJSON('/api/config');
  if (gen !== undefined && gen !== routeGen) return; // superseded: don't paint
  view.innerHTML = '';
  const card = el('div', 'card');
  card.appendChild(el('h2', null, 'Config'));
  const filter = el('input', 'filter'); filter.placeholder = 'filter keys…'; filter.type = 'search';
  card.appendChild(filter);
  const addBtn = el('button', null, 'Add provider…');
  addBtn.style.marginLeft = '8px';
  addBtn.onclick = () => addProvider();
  card.appendChild(addBtn);
  const tbl = el('table', 'cfg');
  const head = el('tr'); ['key','value','source'].forEach(h => head.appendChild(el('th', null, h))); tbl.appendChild(head);
  const keys = Object.keys(data).sort();
  const rows = [];
  for (const k of keys) {
    const s = data[k];
    const tr = el('tr', s.modified ? 'mod' : '');
    tr.appendChild(el('td', 'key', k));
    tr.appendChild(el('td', 'val', s.display)); // the one shared value column (viewmodel.display_value)
    tr.appendChild(el('td', 'src', esc(s.source)));
    // Hover text: the leaf's meaning (the docs table cell), from the same
    // per-key payload the editor overlay shows it in.
    tr.title = s.description ? plainDescription(s.description) : 'click to edit';
    tr.style.cursor = 'pointer';
    actionable(tr, () => editConfig(k, s), 'edit ' + k);
    tbl.appendChild(tr); rows.push([k.toLowerCase(), tr]);
  }
  card.appendChild(tbl);
  filter.oninput = () => { const q = filter.value.toLowerCase(); for (const [key, tr] of rows) tr.style.display = key.includes(q) ? '' : 'none'; };
  view.appendChild(card);
}
// A leaf's description is the docs table cell: markdown-lite. Backtick spans
// render as code; `**bold**` is dropped (the terminal renderers do the same).
function plainDescription(text) { return String(text).replace(/\*\*/g, ''); }
function descriptionNode(text) {
  const frag = document.createDocumentFragment();
  const parts = plainDescription(text).split('`');
  parts.forEach((part, i) => {
    if (!part) return;
    frag.appendChild(i % 2 ? el('code', null, part) : document.createTextNode(part));
  });
  return frag;
}

// The TUI's add-provider form: a whole [providers.<name>] block in one write
// (base_url and auth default from the format and deployment when left blank;
// the key comes from secrets.toml by provider name). Typing a known name
// prefills its format and base_url, as `agent6 connect` would.
async function addProvider() {
  let choices = { api_format: [], deployment: [], defaults: {} };
  try { choices = await getJSON('/api/config/provider_choices'); } catch (e) { toast(e.message, true); return; }
  const back = el('div', 'overlay');
  const opener = document.activeElement;
  const box = el('div', 'card'); box.style.width = 'min(560px, 92vw)';
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', 'add provider');
  const title = el('h2', null, 'Add provider'); title.style.textTransform = 'none';
  box.appendChild(title);
  box.appendChild(el('div', 'sub muted', 'A [providers.<name>] block. base_url and auth default from the format and deployment when left blank.'));
  const labelled = (text, field) => { const l = el('label', 'sub muted', text); l.style.display = 'block'; l.style.marginTop = '8px'; box.appendChild(l); box.appendChild(field); return field; };
  const name = labelled('name', el('input', 'field')); name.placeholder = 'e.g. openrouter, my-azure';
  const select = (values) => { const s = el('select', 'field'); for (const v of values) { const o = el('option', null, v); o.value = v; s.appendChild(o); } return s; };
  const format = labelled('api_format', select(choices.api_format || []));
  const deployment = labelled('deployment', select(choices.deployment || []));
  const baseUrl = labelled('base_url', el('input', 'field')); baseUrl.placeholder = 'blank = default for the format/deployment';
  const keyEnv = labelled('api_key_env', el('input', 'field')); keyEnv.placeholder = 'blank = secrets.toml by provider name';
  let autofilled = '';
  name.oninput = () => {
    const preset = (choices.defaults || {})[name.value.trim()];
    if (!preset) return;
    format.value = preset.api_format;
    if (baseUrl.value === '' || baseUrl.value === autofilled) { autofilled = preset.base_url || ''; baseUrl.value = autofilled; }
  };
  const repoRow = el('label', 'row'); repoRow.style.marginTop = '8px'; repoRow.style.cursor = 'pointer';
  const repoCb = el('input'); repoCb.type = 'checkbox';
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  box.appendChild(repoRow);
  const row = el('div', 'form-row');
  const add = el('button', 'primary', 'Add'), cancel = el('button', null, 'Cancel');
  row.appendChild(add); row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back);
  const close = () => {
    activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey);
    if (opener && opener.isConnected) opener.focus();
  };
  activeOverlayClose = close;
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  cancel.onclick = close;
  back.onclick = (e) => { if (e.target === back) close(); };
  name.focus();
  add.onclick = async () => {
    add.disabled = true;
    try {
      const d = await postJSON('/api/config/provider', {
        name: name.value, api_format: format.value, deployment: deployment.value,
        base_url: baseUrl.value, api_key_env: keyEnv.value, repo: repoCb.checked,
      });
      toast(d.message || 'added'); close(); renderConfig();
    } catch (e) { toast(e.message, true); add.disabled = false; }
  };
}

// A proper editor overlay (not a browser prompt): choices and booleans get a
// select, everything else a text field, with the default, source, and type
// shown; "set for this repo" writes the per-repo config instead of the global.
function editConfig(key, s) {
  const cur = s.value === null || s.value === undefined ? ''
    : Array.isArray(s.value) ? s.value.join(',')
    : typeof s.value === 'object' ? JSON.stringify(s.value)
    : String(s.value);
  const back = el('div', 'overlay');
  const opener = document.activeElement;
  const box = el('div', 'card'); box.style.width = 'min(560px, 92vw)';
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', 'edit ' + key);
  const title = el('h2', null, key);
  title.style.textTransform = 'none'; // a config key is a case-sensitive identifier
  box.appendChild(title);
  const meta = el('div', 'sub muted');
  meta.textContent = `${esc(s.type)} · default: ${s.default_display} · set from: ${esc(s.source)}` + (s.adaptive ? ' · adaptive' : '');
  meta.style.marginBottom = '10px';
  box.appendChild(meta);
  if (s.description) {
    const desc = el('div', 'cfg-desc');
    desc.appendChild(descriptionNode(s.description));
    box.appendChild(desc);
  }
  let field;
  const choices = s.choices || (s.type === 'bool' ? ['true', 'false'] : null);
  if (choices) {
    field = el('select', 'field');
    for (const c of choices) { const o = el('option', null, c); o.value = c; field.appendChild(o); }
    field.value = cur || String(s.default ?? '');
  } else {
    field = el('input', 'field');
    field.value = cur;
    if (s.type === 'list') field.placeholder = 'comma-separated values';
    // Dynamic suggestions (configured provider names, the provider's model
    // ids) from the same sources the TUI config page and CLI TAB completion
    // use, attached as a native datalist autocomplete.
    getJSON('/api/config/suggest/' + encodeURIComponent(key)).then(d => {
      if (!d.values || !d.values.length || !field.isConnected) return;
      const dl = el('datalist'); dl.id = 'cfg-suggest';
      for (const v of d.values) { const o = el('option'); o.value = v; dl.appendChild(o); }
      box.appendChild(dl);
      field.setAttribute('list', dl.id);
    }).catch(() => {});
  }
  box.appendChild(field);
  const repoRow = el('label', 'row'); repoRow.style.marginTop = '8px'; repoRow.style.cursor = 'pointer';
  const repoCb = el('input'); repoCb.type = 'checkbox';
  // Default the target layer to the value's ORIGIN: editing a repo-sourced value
  // must write the repo config, or the repo overlay keeps masking the edit and it
  // looks like the save vanished.
  repoCb.checked = (s.source === 'repo');
  repoRow.appendChild(repoCb); repoRow.appendChild(el('span', 'sub muted', 'set for this repo only (not the global config)'));
  box.appendChild(repoRow);
  const row = el('div', 'form-row');
  const save = el('button', 'primary', 'Save'), cancel = el('button', null, 'Cancel');
  row.appendChild(save);
  // A value set from an editable layer gets an Unset: remove it from that layer
  // so it reverts to the next-lower one / the built-in default. (default-sourced
  // values have nothing to unset; flag/machine layers are not editable here.)
  let unsetBtn = null;
  if (s.source === 'repo' || s.source === 'global') {
    unsetBtn = el('button', null, 'Unset');
    unsetBtn.title = 'remove from the ' + s.source + ' config; reverts to ' + s.default_display;
    row.appendChild(unsetBtn);
  }
  row.appendChild(cancel); box.appendChild(row);
  back.appendChild(box); document.body.appendChild(back);
  const close = () => {
    activeOverlayClose = null; back.remove(); document.removeEventListener('keydown', onKey);
    if (opener && opener.isConnected) opener.focus();
  };
  activeOverlayClose = close; // navigating away dismisses it
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  cancel.onclick = close;
  back.onclick = (e) => { if (e.target === back) close(); };
  field.focus();
  const submit = async () => {
    save.disabled = true;
    try {
      const d = await postJSON('/api/config', { key, value: field.value, repo: repoCb.checked });
      toast(d.message || 'set ' + key); close(); renderConfig();
    } catch (e) { toast(e.message, true); save.disabled = false; }
  };
  save.onclick = submit;
  if (unsetBtn) unsetBtn.onclick = async () => {
    unsetBtn.disabled = true;
    try {
      const d = await postJSON('/api/config', { key, unset: true, repo: s.source === 'repo' });
      toast(d.message || 'unset ' + key); close(); renderConfig();
    } catch (e) { toast(e.message, true); unsetBtn.disabled = false; }
  };
  field.onkeydown = (e) => { if (e.key === 'Enter') { e.preventDefault(); submit(); } };
}
route();
