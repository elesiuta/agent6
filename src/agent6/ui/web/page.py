# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""The web-UI page: HTML + CSS + vanilla JS, served as one string.

Served verbatim by web.server at `GET /`. It renders the wire form the JSON / SSE
endpoints emit (the same shape as `agent6 attach --json`); it is a thin renderer,
so all domain logic stays in the Python read-side.

`client.js` and `styles.css` are real static assets living alongside this
module; read once via `importlib.resources` at import time and spliced in
verbatim, so `PAGE_HTML` stays a module-level constant the server serves
straight from memory and tests can assert against directly.
"""

from __future__ import annotations

from importlib import resources

_ASSETS = resources.files(__package__)
# One script block from the page-family files, concatenated in declaration
# order: the shared core + hub, the run dashboard, the machine watch, config.
_CLIENT_FILES = ("client.js", "client_run.js", "client_machine.js", "client_config.js")
CLIENT_JS = "".join(_ASSETS.joinpath(name).read_text(encoding="utf-8") for name in _CLIENT_FILES)
STYLES_CSS = _ASSETS.joinpath("styles.css").read_text(encoding="utf-8")

# The page is a hash-routed SPA: #/ hub, #/session/<id>, #/machine/<name>,
# #/machines, #/draft/<name>, #/conversation/<id>, #/config. Live views open an
# EventSource against the
# matching /events endpoint; static views fetch a snapshot. Writes are small JSON
# POSTs (new work / steer / approve / answer / merge / prune / config set /
# machine create+run) to the typed endpoints, never arbitrary execution.
PAGE_HTML = (
    r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0e1116">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<title>agent6</title>
<style>
"""
    + STYLES_CSS
    + r"""</style>
</head>
<body>
<aside class="rail">
  <a class="rail-brand" href="#/" aria-label="agent6 home"><img src="/icon.svg" width="24" height="24" alt=""><b>agent6</b></a>
  <nav class="rail-nav">
    <a href="#/" data-tab="hub" title="Sessions"><span class="ic">▤</span><span>Sessions</span></a>
    <a href="#/machines" data-tab="machines" title="Machines"><span class="ic">◈</span><span>Machines</span></a>
    <a href="#/config" data-tab="config" title="Config"><span class="ic">⚙</span><span>Config</span></a>
  </nav>
  <span class="rail-gap"></span>
  <button onclick="toggleTheme()" title="theme">◐<span class="rail-label"> theme</span></button>
  <button id="rail-toggle" onclick="toggleRail()" title="collapse the sidebar"><span id="rail-arrow">«</span><span class="rail-label"> collapse</span></button>
</aside>
<div class="content">
<header>
  <span class="brand" onclick="location.hash='#/'"><b>agent6</b></span>
  <span class="crumb" id="crumb"></span>
  <span class="spacer"></span>
  <button onclick="toggleTheme()" title="theme">◐</button>
</header>

<main id="view"><div class="empty">loading…</div></main>
</div>

<nav class="tabs">
  <a href="#/" data-tab="hub"><span class="ic">▤</span>Sessions</a>
  <a href="#/machines" data-tab="machines"><span class="ic">◈</span>Machines</a>
  <a href="#/config" data-tab="config"><span class="ic">⚙</span>Config</a>
</nav>

<script>
"""
    + CLIENT_JS
    + r"""</script>
</body>
</html>
"""
)


# The PWA manifest: makes the page installable (phone home-screen, desktop app).
# start_url "." keeps it relative to wherever the server is mounted (behind
# `tailscale serve` the path prefix may differ).
MANIFEST_JSON = r"""{
  "name": "agent6",
  "short_name": "agent6",
  "start_url": ".",
  "scope": ".",
  "display": "standalone",
  "background_color": "#0e1116",
  "theme_color": "#0e1116",
  "icons": [
    { "src": "icon.svg", "type": "image/svg+xml", "sizes": "any", "purpose": "any maskable" }
  ]
}
"""

# A minimal service worker: required (with the manifest) for installability. It is
# a network passthrough, no caching, no Web Push / VAPID (OS notifications are the
# foreground Notification API only, fired from the page).
SERVICE_WORKER_JS = r"""self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', () => {});
"""

# The browser-tab favicon: docs/assets/favicon.svg verbatim (keep in sync), so
# the tab shows the same full-bleed glyph as the docs site. The padded ICON_SVG
# below is only for the PWA surfaces, where the safe-area inset is required.
FAVICON_SVG = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" role="img" aria-label="agent6">
  <defs><linearGradient id="g" x1="6" y1="4" x2="42" y2="44" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#7aa2f7"/><stop offset="1" stop-color="#06f5f3"/></linearGradient></defs>
  <path d="M24 3.5 41.7 13.75 V34.25 L24 44.5 6.3 34.25 V13.75 Z" fill="#161618"/>
  <path d="M24 3.5 41.7 13.75 V34.25 L24 44.5 6.3 34.25 V13.75 Z" stroke="url(#g)" stroke-width="2.4" stroke-linejoin="round" fill="none"/>
  <g stroke="url(#g)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <line x1="24.00" y1="24.00" x2="24.00" y2="10.00"/>
    <line x1="24.00" y1="15.32" x2="19.25" y2="12.35"/>
    <line x1="24.00" y1="15.32" x2="28.75" y2="12.35"/>
    <line x1="24.00" y1="24.00" x2="11.88" y2="17.00"/>
    <line x1="16.48" y1="19.66" x2="11.54" y2="22.29"/>
    <line x1="16.48" y1="19.66" x2="16.29" y2="14.06"/>
    <line x1="24.00" y1="24.00" x2="11.88" y2="31.00"/>
    <line x1="16.48" y1="28.34" x2="16.29" y2="33.94"/>
    <line x1="16.48" y1="28.34" x2="11.54" y2="25.71"/>
    <line x1="24.00" y1="24.00" x2="24.00" y2="38.00"/>
    <line x1="24.00" y1="32.68" x2="28.75" y2="35.65"/>
    <line x1="24.00" y1="32.68" x2="19.25" y2="35.65"/>
    <line x1="24.00" y1="24.00" x2="36.12" y2="31.00"/>
    <line x1="31.52" y1="28.34" x2="36.46" y2="25.71"/>
    <line x1="31.52" y1="28.34" x2="31.71" y2="33.94"/>
    <line x1="24.00" y1="24.00" x2="36.12" y2="17.00"/>
    <line x1="31.52" y1="19.66" x2="31.71" y2="14.06"/>
    <line x1="31.52" y1="19.66" x2="36.46" y2="22.29"/>
  </g>
</svg>
"""

# The PWA app icon (manifest + apple-touch): the same snowflake centred on a
# full-bleed dark backdrop so it stays "maskable"-safe. Self-contained SVG, no
# raster asset to ship.
ICON_SVG = r"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs><linearGradient id="g" x1="6" y1="4" x2="42" y2="44" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#7aa2f7"/><stop offset="1" stop-color="#06f5f3"/></linearGradient></defs>
  <rect width="512" height="512" rx="96" fill="#0e1116"/>
  <g transform="translate(96 96) scale(6.6667)">
    <path d="M24 3.5 41.7 13.75 V34.25 L24 44.5 6.3 34.25 V13.75 Z" fill="#161618"/>
    <path d="M24 3.5 41.7 13.75 V34.25 L24 44.5 6.3 34.25 V13.75 Z" stroke="url(#g)" stroke-width="2.4" stroke-linejoin="round" fill="none"/>
    <g stroke="url(#g)" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" fill="none">
      <line x1="24" y1="24" x2="24" y2="10"/>
      <line x1="24" y1="15.32" x2="19.25" y2="12.35"/>
      <line x1="24" y1="15.32" x2="28.75" y2="12.35"/>
      <line x1="24" y1="24" x2="11.88" y2="17"/>
      <line x1="16.48" y1="19.66" x2="11.54" y2="22.29"/>
      <line x1="16.48" y1="19.66" x2="16.29" y2="14.06"/>
      <line x1="24" y1="24" x2="11.88" y2="31"/>
      <line x1="16.48" y1="28.34" x2="16.29" y2="33.94"/>
      <line x1="16.48" y1="28.34" x2="11.54" y2="25.71"/>
      <line x1="24" y1="24" x2="24" y2="38"/>
      <line x1="24" y1="32.68" x2="28.75" y2="35.65"/>
      <line x1="24" y1="32.68" x2="19.25" y2="35.65"/>
      <line x1="24" y1="24" x2="36.12" y2="31"/>
      <line x1="31.52" y1="28.34" x2="36.46" y2="25.71"/>
      <line x1="31.52" y1="28.34" x2="31.71" y2="33.94"/>
      <line x1="24" y1="24" x2="36.12" y2="17"/>
      <line x1="31.52" y1="19.66" x2="31.71" y2="14.06"/>
      <line x1="31.52" y1="19.66" x2="36.46" y2="22.29"/>
    </g>
  </g>
</svg>
"""
