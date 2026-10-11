"""KONAI 감사 테스트 공통 로더.
GUI/외부 의존성(tkinter, customtkinter, win32gui, FinanceDataReader, OpenDartReader 등)을 모의 모듈로 대체하고
임시 작업 디렉터리에서 소스를 import 한다. 실제 캐시·설정·네트워크는 건드리지 않는다.
소스 경로: 환경변수 KONAI_SRC (기본: 저장소 루트의 konai_v9.1.99.py)."""
import importlib.util
import os
import sys
import tempfile
from unittest import mock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT = os.path.join(os.path.dirname(_HERE), "konai_v9.1.99.py")


@pytest.fixture(scope="session")
def k():
    src = os.environ.get("KONAI_SRC", _DEFAULT)
    if not os.path.exists(src):
        pytest.skip(f"소스 없음: {src} (KONAI_SRC 지정 필요)")
    tmp = tempfile.mkdtemp(prefix="konai_t_")
    old = os.getcwd()
    os.environ["HOME"] = tmp
    os.environ["USERPROFILE"] = tmp
    os.chdir(tmp)
    for name in ["FinanceDataReader", "OpenDartReader", "win32gui", "pyperclip", "pyautogui", "yfinance",
                 "pykrx", "customtkinter", "tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.font"]:
        sys.modules[name] = mock.MagicMock()

    class _Base:
        def __init__(self, *a, **kw):
            pass
    sys.modules["customtkinter"].CTk = _Base
    spec = importlib.util.spec_from_file_location("konai_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["konai_under_test"] = mod
    spec.loader.exec_module(mod)
    yield mod
    os.chdir(old)
