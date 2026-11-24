"""Regression tests for repository-wide credential, .env and network isolation."""
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys

import dotenv
import dotenv.main
import pytest

from src import gemini as gemini_module
from src.gemini import DEFAULT_MODEL, Gemini

ROOT = Path(__file__).resolve().parents[1]
ISOLATED = ("GEMINI_API_KEY", "GEMINI_MODEL", "DOCGROUND_FREE_TIER_CONFIRMED",
            "GOOGLE_API_KEY", "GOOGLE_GENAI_HTTP_CLIENT")
BLOCKED = "may not open external network connections"
# Loaded before the repository fixtures in probe runs. It replaces the network entry points the guard
# wraps, so a probe can never send real traffic and any call that gets past the guard is logged.
BOUNDARY_PLUGIN = '''
import os
import socket


def _stub(name):
    def stub(*args, **kwargs):
        with open(os.environ["DOCGROUND_BOUNDARY_LOG"], "a", encoding="utf-8") as log:
            log.write(name + "\\n")
        raise OSError("stubbed network boundary")
    return stub


for _name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
    setattr(socket, _name, _stub(_name))
for _name in ("connect", "connect_ex", "sendto"):
    setattr(socket.socket, _name, _stub("socket." + _name))
'''


def _run_probe(tmp_path, source, *, isolated=True, extra_env=None):
    """Run a probe module in a fresh pytest process behind the stubbed network boundary.

    Returns the completed process and the network entry points that the probe reached.
    """
    (tmp_path / "docground_boundary.py").write_text(BOUNDARY_PLUGIN, encoding="utf-8")
    probe = tmp_path / "test_probe.py"
    probe.write_text(source, encoding="utf-8")
    log = tmp_path / "boundary.log"
    log.unlink(missing_ok=True)
    env = os.environ | (extra_env or {}) | {
        "DOCGROUND_BOUNDARY_LOG": str(log),
        "PYTHONPATH": os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))),
    }
    env.pop("PYTHON_DOTENV_DISABLED", None)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
               "-p", "docground_boundary"]
    if isolated:
        command += ["-p", "tests.conftest"]
    result = subprocess.run(command + [str(probe)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    reached = log.read_text(encoding="utf-8").split() if log.exists() else []
    return result, reached


def test_constructing_gemini_does_not_load_dotenv(monkeypatch):
    calls = []

    def record_load(*args, **kwargs):
        calls.append("load_dotenv")
        return False

    def record_find(*args, **kwargs):
        calls.append("find_dotenv")
        return ""

    # Re-enable the library path so a regressed implicit load would reach find_dotenv and be recorded.
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED")
    monkeypatch.setattr(dotenv, "load_dotenv", record_load)
    monkeypatch.setattr(dotenv.main, "load_dotenv", record_load)
    monkeypatch.setattr(dotenv.main, "find_dotenv", record_find)
    provider = Gemini()
    assert calls == []
    assert provider.model == DEFAULT_MODEL and "GEMINI_MODEL" not in os.environ
    assert not hasattr(gemini_module, "load_dotenv")


def test_dotenv_loading_is_disabled_for_explicit_paths_and_streams(tmp_path):
    local = tmp_path / ".env"
    local.write_text("GEMINI_API_KEY=fixture-from-dotenv\n", encoding="utf-8")
    try:
        assert os.environ["PYTHON_DOTENV_DISABLED"] == "1"
        assert dotenv.load_dotenv(local) is False
        with local.open(encoding="utf-8") as stream:
            assert dotenv.load_dotenv(stream=stream) is False
        assert "GEMINI_API_KEY" not in os.environ
    finally:
        os.environ.pop("GEMINI_API_KEY", None)


def test_credentials_are_isolated_from_module_import_through_each_test(tmp_path):
    probe = (
        "import os\n\n"
        f"NAMES = {ISOLATED!r}\n"
        "AT_IMPORT = {'credentials': [name for name in NAMES if name in os.environ],\n"
        "             'dotenv_disabled': os.environ.get('PYTHON_DOTENV_DISABLED')}\n\n\n"
        "def test_import_state():\n"
        "    assert AT_IMPORT == {'credentials': [], 'dotenv_disabled': '1'}\n\n\n"
        "def test_direct_write():\n"
        "    os.environ['GEMINI_API_KEY'] = 'written-directly-by-a-test'\n\n\n"
        "def test_next_test_starts_clean():\n"
        "    assert [name for name in NAMES if name in os.environ] == []\n"
        "    assert os.environ['PYTHON_DOTENV_DISABLED'] == '1'\n"
    )
    inherited = {name: "synthetic-inherited-value" for name in ISOLATED}
    control, _ = _run_probe(tmp_path, probe, isolated=False, extra_env=inherited)
    assert control.returncode == 1 and "2 failed, 1 passed" in control.stdout, control.stdout
    result, reached = _run_probe(tmp_path, probe, extra_env=inherited)
    assert result.returncode == 0 and "3 passed" in result.stdout, result.stdout + result.stderr
    assert reached == []


def test_network_access_while_importing_tests_fails_the_run(tmp_path):
    probe = (
        "import socket\n\n"
        "try:\n"
        "    socket.gethostbyname('example.com')\n"
        "except Exception:\n"
        "    pass\n"
        "with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:\n"
        "    try:\n"
        "        udp.sendto(b'x', ('192.0.2.1', 9))\n"
        "    except Exception:\n"
        "        pass\n\n\n"
        "def test_probe():\n"
        "    pass\n"
    )
    control, reached = _run_probe(tmp_path, probe, isolated=False)
    assert control.returncode == 0 and reached == ["gethostbyname", "socket.sendto"]
    result, reached = _run_probe(tmp_path, probe)
    assert result.returncode == 1 and reached == []
    assert "External network access attempted while importing tests" in result.stdout + result.stderr


def test_swallowed_external_attempt_still_fails_the_test(tmp_path):
    probe = (
        "import socket\n\n\n"
        "def test_probe():\n"
        "    try:\n"
        "        socket.getaddrinfo('example.com', 443)\n"
        "    except Exception:\n"
        "        pass\n"
    )
    result, reached = _run_probe(tmp_path, probe)
    assert result.returncode == 1 and "External network access attempted" in result.stdout
    assert reached == []


def _call(function, *args):
    def run(boundary):
        with boundary:
            return function(*args)
    return run


def _socket_call(family, kind, method, *args):
    def call():
        with socket.socket(family, kind) as sock:
            sock.settimeout(1)
            return getattr(sock, method)(*args)
    return _call(call)


def _event_loop_call(family, kind, method, address):
    def run(boundary):
        # The loop builds its self-pipe over real loopback, so it is created before the boundary closes.
        loop = asyncio.new_event_loop()
        try:
            with socket.socket(family, kind) as sock:
                sock.setblocking(False)
                operation = (loop.sock_sendto(sock, b"x", address) if method == "sock_sendto"
                             else loop.sock_connect(sock, address))
                with boundary:
                    return loop.run_until_complete(operation)
        finally:
            loop.close()
    return run


TEST_NET = ("192.0.2.1", 9)
TEST_NET_V6 = ("2001:db8::1", 9)
BLOCKED_CALLS = {
    "getaddrinfo": _call(socket.getaddrinfo, "example.com", 443),
    "create_connection": _call(socket.create_connection, ("example.com", 443), 1),
    "gethostbyname": _call(socket.gethostbyname, "example.com"),
    "gethostbyname_ex": _call(socket.gethostbyname_ex, "example.com"),
    "gethostbyaddr": _call(socket.gethostbyaddr, "192.0.2.1"),
    "getnameinfo": _call(socket.getnameinfo, ("192.0.2.1", 80), 0),
    "getfqdn": _call(socket.getfqdn, "example.com"),
    "tcp_connect": _socket_call(socket.AF_INET, socket.SOCK_STREAM, "connect", TEST_NET),
    "tcp_connect_ex": _socket_call(socket.AF_INET, socket.SOCK_STREAM, "connect_ex", TEST_NET),
    "tcp_connect_ipv6": _socket_call(socket.AF_INET6, socket.SOCK_STREAM, "connect", TEST_NET_V6),
    "udp_connect": _socket_call(socket.AF_INET, socket.SOCK_DGRAM, "connect", TEST_NET),
    "udp_sendto": _socket_call(socket.AF_INET, socket.SOCK_DGRAM, "sendto", b"x", TEST_NET),
    "udp_sendto_flags": _socket_call(socket.AF_INET, socket.SOCK_DGRAM, "sendto", b"x", 0, TEST_NET),
    "udp_sendto_ipv6": _socket_call(socket.AF_INET6, socket.SOCK_DGRAM, "sendto", b"x", TEST_NET_V6),
    "udp_sendmsg": _socket_call(socket.AF_INET, socket.SOCK_DGRAM, "sendmsg", [b"x"], [], 0, TEST_NET),
    "event_loop_connect": _event_loop_call(socket.AF_INET, socket.SOCK_STREAM, "sock_connect", TEST_NET),
    "event_loop_sendto": _event_loop_call(socket.AF_INET, socket.SOCK_DGRAM, "sock_sendto", TEST_NET),
}


@pytest.mark.parametrize("name", BLOCKED_CALLS)
def test_external_network_paths_are_blocked_before_the_boundary(network_boundary, blocked_network_attempts, name):
    if name == "udp_sendmsg" and not hasattr(socket.socket, "sendmsg"):
        pytest.skip("socket.sendmsg is unavailable on this platform")
    start = len(blocked_network_attempts)
    with pytest.raises(RuntimeError, match=BLOCKED):
        BLOCKED_CALLS[name](network_boundary)
    assert network_boundary.reached == []
    assert len(blocked_network_attempts) == start + 1


@pytest.mark.parametrize(("name", "call"), [
    ("getaddrinfo", lambda: socket.getaddrinfo("localhost", 80)),
    ("gethostbyname", lambda: socket.gethostbyname("localhost")),
    ("gethostbyname_ex", lambda: socket.gethostbyname_ex("127.0.0.1")),
    ("gethostbyaddr", lambda: socket.gethostbyaddr("::1")),
    ("getnameinfo", lambda: socket.getnameinfo(("127.0.0.1", 80), 0)),
])
def test_loopback_lookups_pass_through_the_guard(network_boundary, name, call):
    # The stub stands in for the real resolver, so allowed lookups are proven without DNS traffic.
    with network_boundary as reached:
        assert call() == "boundary-stub"
    assert reached == [name]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_tcp_connections_still_work(host):
    family, bind = (socket.AF_INET6, "::1") if host == "::1" else (socket.AF_INET, "127.0.0.1")
    try:
        server = socket.create_server((bind, 0), family=family)
    except OSError:
        pytest.skip(f"{bind} loopback is unavailable on this host")
    with server:
        port = server.getsockname()[1]
        with socket.create_connection((host, port), timeout=5) as client:
            peer, _ = server.accept()
            with peer:
                client.sendall(b"ok")
                assert peer.recv(2) == b"ok"


@pytest.mark.parametrize(("family", "host"), [(socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")])
def test_loopback_udp_sends_still_work(family, host):
    with socket.socket(family, socket.SOCK_DGRAM) as receiver, socket.socket(family, socket.SOCK_DGRAM) as sender:
        try:
            receiver.bind((host, 0))
        except OSError:
            pytest.skip(f"{host} loopback is unavailable on this host")
        receiver.settimeout(5)
        sender.sendto(b"ok", receiver.getsockname())
        assert receiver.recvfrom(2)[0] == b"ok"


def test_event_loop_loopback_connect_and_datagram_still_work():
    async def exchange(server_address, receiver_address):
        loop = asyncio.get_running_loop()
        with socket.socket() as client, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            client.setblocking(False)
            sender.setblocking(False)
            await loop.sock_connect(client, server_address)
            await loop.sock_sendto(sender, b"ok", receiver_address)

    with socket.create_server(("127.0.0.1", 0)) as server, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5)
        asyncio.run(exchange(server.getsockname(), receiver.getsockname()))
        peer, _ = server.accept()
        peer.close()
        assert receiver.recvfrom(2)[0] == b"ok"


def test_socketpair_and_asyncio_event_loop_still_work():
    left, right = socket.socketpair()
    with left, right:
        left.sendall(b"x")
        assert right.recv(1) == b"x"
    # On Windows the default Proactor loop creates its self-pipe through a loopback socket pair.
    assert asyncio.run(asyncio.sleep(0, result="done")) == "done"
