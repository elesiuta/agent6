# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Eric Lesiuta
"""Read one URL, for a worker whose commands have no network.

Under the default `network`, a jailed command has no network at all, so
the worker cannot read a linked spec, an API's docs or a changelog. Its only
move is to ask the operator and wait. This runs in the AGENT process, which
already has egress, and hands the bytes back as a tool result.

Not a crawler and not a client: one URL, GET only, no redirects followed, no
header or body the model chose, and no credential ever sent. Every refusal is
a default-deny -- a scheme that is not https, an address that is not global, a
body that is not text -- rather than a list of bad things.

It is still an egress channel a model drives: a GET can encode data in its
path. That is why a host is either on the operator's allow-list or asked
about, and why an absent operator is a no.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx2

from agent6.tools.http_body import BodyRefused, read_capped

# A body is context the operator pays for, and a fetch is meant to answer one
# question. Beyond this the read is refused, never silently truncated.
MAX_BYTES = 1 << 20
TIMEOUT_S = 20.0
# What a model can read: prose and structured data. A binary blob in the
# context window is noise, so it is refused by what it IS, not by extension.
_TEXTUAL = ("text/", "application/json", "application/xml", "application/xhtml+xml")
# The allow-list value that means "any host". Empty means NONE, so opting out
# has to be written down and shows up in `agent6 config show` as a choice.
ANY_HOST = "*"


class FetchRefused(Exception):
    """The URL was not fetched, and why."""


@dataclass(frozen=True, slots=True)
class Fetched:
    """One URL's response."""

    url: str
    status: int
    content_type: str
    body: str
    # A 30x's target, handed back for the model to decide on rather than
    # followed.
    location: str = ""


def host_allowed(host: str, allowed: tuple[str, ...]) -> bool:
    """Whether *host* is on the operator's list.

    Hosts, never URL prefixes: a prefix invites `evil.com/docs.python.org`. A
    leading dot allows subdomains, so `.readthedocs.io` covers the project
    pages without covering `notreadthedocs.io`.
    """
    if ANY_HOST in allowed:
        return True
    host = host.lower().rstrip(".")
    for entry in allowed:
        pattern = entry.lower().rstrip(".")
        if pattern.startswith("."):
            if host == pattern[1:] or host.endswith(pattern):
                return True
        elif host == pattern:
            return True
    return False


@dataclass(frozen=True, slots=True)
class Checked:
    """A vetted URL that has not touched the network: the gate's input."""

    url: str
    host: str

    def prompt(self) -> str:
        """The approval line: the parsed host (never the raw URL, so the name
        shown is exactly the one the connection is proved against and the one
        `fetch_hosts` would have to name), a port other than 443, then the
        full path and query. A GET
        carries data out in its query string, so clipping the path or dropping
        the query is consent to an exfiltration the operator never saw."""
        parts = urlsplit(self.url)
        tail = parts.path or "/"
        if parts.query:
            tail += f"?{parts.query}"
        port = "" if parts.port in (None, 443) else f":{parts.port}"
        return f"{self.host}{port} {tail}"


def check_url(url: str) -> Checked:
    """Vet *url* without touching the network, or raise.

    Everything the string alone can prove: https, no credentials, a real host,
    and a literal address that is public. A name is NOT resolved here: the DNS
    query for `<data>.attacker.example` delivers its label to whoever runs
    that name's authoritative server, so resolving ahead of the operator's
    gate was itself an egress channel. `fetch` resolves behind the gate.
    """
    try:
        parts = urlsplit(url)
        _ = parts.port  # a port outside 0-65535 raises here, before the approval
    except ValueError as exc:
        # urlsplit refuses a malformed literal ("http://[::1") and the port
        # accessor an out-of-range port; as a FetchRefused either reads like
        # every other fetch refusal instead of the dispatcher's "failed:".
        raise FetchRefused(f"the URL cannot be read: {exc}") from exc
    if parts.scheme != "https":
        raise FetchRefused(f"only https is fetched, not {parts.scheme or 'a bare path'!r}")
    if parts.username or parts.password:
        # httpx turns userinfo into an Authorization header, so this is the
        # model choosing a credential as well as disguising the real host.
        raise FetchRefused("a URL with credentials in it is not fetched")
    host = parts.hostname
    if not host:
        raise FetchRefused("no host in the URL")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        return Checked(url=url, host=host)  # a name: resolved behind the gate
    if not literal.is_global:
        raise FetchRefused(f"{host} is not a public address")
    return Checked(url=url, host=host)


def fetch(checked: Checked) -> Fetched:
    """GET *checked*, refusing anything that is not a bounded text response.

    The host resolves HERE, behind every gate, to public addresses only --
    which keeps a fetch away from the cloud metadata endpoint
    (169.254.169.254), a loopback admin port, or the operator's LAN. The
    connection then dials exactly the address chosen, with the original name
    in SNI and Host, so the certificate is still proved against the name while
    no second DNS answer can move it. Handing the name onward instead let two
    resolvers disagree: CPython's `getaddrinfo` encodes an international name
    with IDNA2003 and httpx with UTS-46, so `ßeta.example.com` was vetted as
    `sseta.example.com` and connected to `xn--eta-4ka.example.com` -- a
    different host entirely, and a complete bypass needing no race at all.
    Re-resolving also reopened the plain rebinding window.

    Redirects are returned, not followed: a 30x hands its Location back for the
    model to decide on, which re-runs every check. Following them silently is
    how one allowed host becomes an open proxy to every other.
    """
    parts = urlsplit(checked.url)
    port = 443 if parts.port is None else parts.port
    try:
        infos = socket.getaddrinfo(checked.host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise FetchRefused(f"{checked.host} does not resolve: {exc}") from exc
    address = ""
    for info in infos:
        addr = ipaddress.ip_address(str(info[4][0]))
        if not addr.is_global:
            raise FetchRefused(f"{checked.host} resolves to {addr}, which is not a public address")
        address = address or str(addr)
    if not address:
        raise FetchRefused(f"{checked.host} resolves to nothing")
    literal = f"[{address}]" if ":" in address else address
    dialled = parts._replace(netloc=f"{literal}:{port}").geturl()
    try:
        with (
            httpx2.Client(follow_redirects=False, timeout=TIMEOUT_S, verify=True) as client,
            client.stream(
                "GET",
                dialled,
                # Compression is declined here and refused below if the server
                # sends it anyway: the cap counts what ARRIVES, and a decoded
                # stream would expand past it before any check.
                headers={"Host": checked.host, "Accept-Encoding": "identity"},
                extensions={"sni_hostname": checked.host},
            ) as response,
        ):
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith(_TEXTUAL):
                raise FetchRefused(f"not a text response: content-type {content_type!r}")
            deadline = time.monotonic() + TIMEOUT_S
            body = read_capped(response, cap=MAX_BYTES, deadline=deadline, timeout_s=TIMEOUT_S)
            return Fetched(
                url=checked.url,
                status=response.status_code,
                content_type=content_type,
                body=body.decode(response.encoding or "utf-8", errors="replace"),
                location=response.headers.get("location", ""),
            )
    except BodyRefused as exc:
        raise FetchRefused(str(exc)) from exc
    except httpx2.HTTPError as exc:
        raise FetchRefused(f"could not fetch {checked.url}: {exc}") from exc
