"""Repository-wide test isolation: no local credentials and no external network.

Isolation starts in ``pytest_configure``, before pytest imports any test module, and is undone in
``pytest_unconfigure``. Each test also starts and ends with the provider credentials removed.

Guarded outbound paths (loopback and AF_UNIX stay available):

- ``socket.socket.connect`` and ``connect_ex`` (TCP and connected UDP);
- ``socket.socket.sendto`` and, where the platform has it, ``sendmsg`` with a destination (UDP);
- ``socket.getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``, ``gethostbyaddr`` and
  ``getnameinfo`` (Python DNS lookups). ``socket.create_connection`` and ``socket.getfqdn`` call these;
- On Windows, ``asyncio.windows_events.IocpProactor.connect`` and ``sendto``. The default Proactor
  event loop sends through overlapped I/O and never calls the socket methods above.

Remaining limits: native extensions that open their own sockets, direct use of the private
``_socket`` module, and functions bound with ``from socket import ...`` before ``pytest_configure``
bypass the guard. Raw ``send`` on an unconnected datagram socket has no destination to inspect.
``socket.socketpair`` is untouched, so Streamlit and asyncio loopback plumbing keep working.
"""
import ipaddress
import os
import socket
import sys

import pytest

ISOLATED_VARIABLES = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "DOCGROUND_FREE_TIER_CONFIRMED",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_HTTP_CLIENT",
    "GROQ_API_KEY",
    "DOCGROUND_GENERATION_PROVIDER",
)
DOTENV_SWITCH = "PYTHON_DOTENV_DISABLED"
_ALLOW = object()
_BLOCKED_ATTEMPTS: list[tuple[str, object]] = []
# Real network entry points, captured when the guard is installed. Tests may stub these.
_REAL: dict[str, object] = {}
_SESSION_PATCH: pytest.MonkeyPatch | None = None


class ExternalNetworkBlocked(RuntimeError):
    """Raised for non-loopback traffic. Not an OSError, so callers cannot treat it as an outage."""


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host is None or host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _destination(family, address):
    unix = getattr(socket, "AF_UNIX", None)
    if address is None or (unix is not None and family == unix):
        return _ALLOW
    if isinstance(address, tuple) and address and _is_loopback(address[0]):
        return _ALLOW
    return address


# Each checker receives the original call arguments and returns _ALLOW or the refused target.
_SOCKET_METHODS = {
    "connect": lambda sock, address, *rest: _destination(sock.family, address),
    "connect_ex": lambda sock, address, *rest: _destination(sock.family, address),
    "sendto": lambda sock, data, *rest: _destination(sock.family, rest[-1] if rest else None),
    "sendmsg": lambda sock, buffers, *rest: _destination(sock.family, rest[2] if len(rest) > 2 else None),
}
_RESOLVERS = {
    "getaddrinfo": lambda host, *rest, **kwargs: _ALLOW if _is_loopback(host) else host,
    "gethostbyname": lambda host: _ALLOW if _is_loopback(host) else host,
    "gethostbyname_ex": lambda host: _ALLOW if _is_loopback(host) else host,
    "gethostbyaddr": lambda host: _ALLOW if _is_loopback(host) else host,
    "getnameinfo": lambda sockaddr, flags: _destination(None, sockaddr),
}
_PROACTOR_METHODS = {
    "connect": lambda proactor, conn, address: _destination(conn.family, address),
    "sendto": lambda proactor, conn, buf, flags=0, addr=None: _destination(conn.family, addr),
}


def _guarded(key, kind, checker):
    def guard(*args, **kwargs):
        target = checker(*args, **kwargs)
        if target is not _ALLOW:
            _BLOCKED_ATTEMPTS.append((kind, target))
            raise ExternalNetworkBlocked(f"Tests may not open external network connections ({kind} {target!r}).")
        return _REAL[key](*args, **kwargs)
    return guard


def _guard_targets():
    targets = [(socket.socket, name, f"socket.{name}", name, checker)
               for name, checker in _SOCKET_METHODS.items() if hasattr(socket.socket, name)]
    targets += [(socket, name, name, name, checker) for name, checker in _RESOLVERS.items()]
    if sys.platform == "win32":
        from asyncio.windows_events import IocpProactor
        targets += [(IocpProactor, name, f"proactor.{name}", f"proactor_{name}", checker)
                    for name, checker in _PROACTOR_METHODS.items()]
    return targets


def _install_guard(patch: pytest.MonkeyPatch) -> None:
    for owner, attribute, key, kind, checker in _guard_targets():
        _REAL[key] = getattr(owner, attribute)
        patch.setattr(owner, attribute, _guarded(key, kind, checker))


def _scrub_environment() -> None:
    for name in ISOLATED_VARIABLES:
        os.environ.pop(name, None)
    # python-dotenv honors this switch for every load_dotenv call, however it was imported.
    os.environ[DOTENV_SWITCH] = "1"


def pytest_configure(config):
    """Isolate before collection imports test modules or the application code they import."""
    global _SESSION_PATCH
    if _SESSION_PATCH is not None:
        return
    _SESSION_PATCH = pytest.MonkeyPatch()
    for name in ISOLATED_VARIABLES:
        _SESSION_PATCH.delenv(name, raising=False)
    _SESSION_PATCH.setenv(DOTENV_SWITCH, "1")
    _install_guard(_SESSION_PATCH)


def pytest_unconfigure(config):
    global _SESSION_PATCH
    if _SESSION_PATCH is not None:
        _SESSION_PATCH.undo()
        _SESSION_PATCH = None
        _REAL.clear()


def pytest_collection_finish(session):
    """A module that touches the network while being imported fails the whole run."""
    if _BLOCKED_ATTEMPTS:
        attempts = list(_BLOCKED_ATTEMPTS)
        _BLOCKED_ATTEMPTS.clear()
        pytest.exit(f"External network access attempted while importing tests: {attempts}", returncode=1)


@pytest.fixture(autouse=True)
def _fail_on_external_network():
    """Fail a test whose code swallowed a blocked attempt instead of surfacing it."""
    start = len(_BLOCKED_ATTEMPTS)
    yield
    attempts = _BLOCKED_ATTEMPTS[start:]
    del _BLOCKED_ATTEMPTS[start:]
    if attempts:
        pytest.fail(f"External network access attempted: {attempts}")


@pytest.fixture
def blocked_network_attempts():
    """For tests that deliberately attempt external access. Consumes the recorded attempts."""
    start = len(_BLOCKED_ATTEMPTS)
    yield _BLOCKED_ATTEMPTS
    del _BLOCKED_ATTEMPTS[start:]


class NetworkBoundary:
    """While entered, every real network entry point is a recorder, so no traffic can leave.

    Use it around the call under test only. Event loop construction needs real loopback sockets.
    """

    def __init__(self):
        self.reached: list[str] = []
        self._patch = pytest.MonkeyPatch()

    def _stub(self, key):
        def stub(*args, **kwargs):
            self.reached.append(key)
            return "boundary-stub"
        return stub

    def __enter__(self):
        for key in list(_REAL):
            self._patch.setitem(_REAL, key, self._stub(key))
        return self.reached

    def __exit__(self, *exc_info):
        self._patch.undo()


@pytest.fixture
def network_boundary():
    boundary = NetworkBoundary()
    yield boundary
    boundary._patch.undo()


@pytest.fixture(autouse=True)
def _isolated_environment():
    """Each test starts and ends without provider credentials and with .env loading disabled.

    Cleanup does not depend on monkeypatch bookkeeping, so direct os.environ writes cannot leak.
    """
    _scrub_environment()
    yield
    _scrub_environment()
