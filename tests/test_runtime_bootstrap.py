from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

from live_translate.runtime_bootstrap import configure_runtime_paths, ensure_runtime_stdlib_shims


def test_ensure_runtime_stdlib_shims_copies_required_stdlib_modules(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime_python = root / "runtime_libraries" / "python"
        runtime_python.mkdir(parents=True)
        portable_lib = root / "python312" / "Lib"
        portable_lib.mkdir(parents=True)
        (portable_lib / "pickletools.py").write_text("# pickletools shim\n", encoding="utf-8")
        (portable_lib / "timeit.py").write_text("# timeit shim\n", encoding="utf-8")
        (portable_lib / "wave.py").write_text("# wave shim\n", encoding="utf-8")
        (portable_lib / "chunk.py").write_text("# chunk shim\n", encoding="utf-8")
        (portable_lib / "symtable.py").write_text("# symtable shim\n", encoding="utf-8")

        monkeypatch.setattr(
            "live_translate.runtime_bootstrap._runtime_library_root",
            lambda: root / "runtime_libraries",
        )
        monkeypatch.setattr(
            "live_translate.runtime_dependencies.portable_app_root",
            lambda: root,
        )

        ensure_runtime_stdlib_shims()

        copied_pickletools = runtime_python / "pickletools.py"
        copied_timeit = runtime_python / "timeit.py"
        copied_wave = runtime_python / "wave.py"
        copied_chunk = runtime_python / "chunk.py"
        copied_symtable = runtime_python / "symtable.py"
        assert copied_pickletools.is_file()
        assert copied_timeit.is_file()
        assert copied_wave.is_file()
        assert copied_chunk.is_file()
        assert copied_symtable.is_file()
        assert copied_pickletools.read_text(encoding="utf-8") == "# pickletools shim\n"
        assert copied_timeit.read_text(encoding="utf-8") == "# timeit shim\n"
        assert copied_wave.read_text(encoding="utf-8") == "# wave shim\n"
        assert copied_chunk.read_text(encoding="utf-8") == "# chunk shim\n"
        assert copied_symtable.read_text(encoding="utf-8") == "# symtable shim\n"


def test_configure_runtime_paths_releases_shadowed_packaging_modules(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime_python = root / "runtime_libraries" / "python"
        runtime_python.mkdir(parents=True)
        bundled_setuptools = root / "_internal" / "setuptools" / "__init__.py"
        bundled_setuptools.parent.mkdir(parents=True)
        bundled_setuptools.write_text("", encoding="utf-8")

        module = types.ModuleType("setuptools")
        module.__file__ = str(bundled_setuptools)
        sys.modules["setuptools"] = module
        distutils_module = types.ModuleType("distutils")
        distutils_module.__file__ = str(root / "_internal" / "distutils" / "__init__.py")
        sys.modules["distutils"] = distutils_module
        xml_module = types.ModuleType("xml")
        xml_module.__file__ = str(root / "_internal" / "xml" / "__init__.py")
        xml_module.__path__ = [str(root / "_internal" / "xml")]
        sys.modules["xml"] = xml_module

        monkeypatch.setattr(
            "live_translate.runtime_bootstrap._runtime_library_root",
            lambda: root / "runtime_libraries",
        )
        monkeypatch.setattr(
            "live_translate.runtime_dependencies.portable_app_root",
            lambda: root,
        )

        try:
            configure_runtime_paths()

            assert "setuptools" not in sys.modules
            assert "distutils" not in sys.modules
            assert "xml" not in sys.modules
            assert sys.path[0] == str(runtime_python)
        finally:
            sys.modules.pop("setuptools", None)
            sys.modules.pop("distutils", None)
            sys.modules.pop("xml", None)


def test_ensure_runtime_stdlib_shims_copies_distutils_from_setuptools(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        runtime_python = root / "runtime_libraries" / "python"
        source_command = runtime_python / "setuptools" / "_distutils" / "command"
        source_command.mkdir(parents=True)
        (source_command / "build_scripts.py").write_text("# build scripts\n", encoding="utf-8")
        portable_lib = root / "python312" / "Lib"
        portable_lib.mkdir(parents=True)

        monkeypatch.setattr(
            "live_translate.runtime_bootstrap._runtime_library_root",
            lambda: root / "runtime_libraries",
        )
        monkeypatch.setattr(
            "live_translate.runtime_dependencies.portable_app_root",
            lambda: root,
        )

        ensure_runtime_stdlib_shims()

        copied = runtime_python / "distutils" / "command" / "build_scripts.py"
        assert copied.read_text(encoding="utf-8") == "# build scripts\n"
