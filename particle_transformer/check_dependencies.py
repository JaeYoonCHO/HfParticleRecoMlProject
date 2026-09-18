"""Check training dependencies without reading data or changing the environment.

Run with the Python environment to inspect, for example:
    TransfEnv/bin/python check_dependencies.py

Reference versions passed a local CPU model smoke test. They are not minimum
requirements: the training sources do not specify supported version ranges.
"""

import importlib
from importlib import metadata
import platform
import subprocess
import sys


DEPENDENCIES = (
    ("torch", "torch", "2.14.0"),
    ("numpy", "numpy", "2.5.3"),
    ("pandas", "pandas", "3.0.5"),
    ("uproot", "uproot", "5.7.6"),
    ("scikit-learn", "sklearn", "1.9.0"),
    ("PyYAML", "yaml", "6.0.3"),
    ("tqdm", "tqdm", "4.70.0"),
)


def main():
    print("Python:", platform.python_version(), "(tested reference: 3.12.9)")
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())
    print("Virtual environment:", sys.prefix != sys.base_prefix)
    print("Required version ranges: unspecified in the training sources.")
    print("Reference versions are tested versions, not minimum requirements.")
    print("Checking imports; first-time imports can take a while.", flush=True)
    print()
    print(f"{'Package':<16} {'Reference':<14} {'Installed':<18} Status")
    print("-" * 76)

    errors = []
    torch_module = None
    for distribution, module_name, reference in DEPENDENCIES:
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            installed = "missing"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            status = "FAIL"
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")
        else:
            if installed == "missing":
                status = "FAIL (metadata missing)"
                errors.append(f"{distribution}: import succeeded but package metadata is missing")
            else:
                status = "OK" if installed == reference else "OK (different version)"
            if module_name == "torch":
                torch_module = module
        print(f"{distribution:<16} {reference:<14} {installed:<18} {status}", flush=True)

    if torch_module is not None:
        print("\nCUDA available:", torch_module.cuda.is_available())
        mps = getattr(torch_module.backends, "mps", None)
        print("MPS available:", bool(mps and mps.is_available()))
        print("Training device=auto selects CUDA, then MPS, then CPU.")

    print("\nChecking transitive dependency constraints with pip check...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "check"],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip() or result.stderr.strip() or "No pip output.")
    if result.returncode:
        errors.append("pip check failed; see output above")

    print("\nStandard-library modules need no separate installation.")
    print("ROOT files are read via uproot; PyROOT/CERN ROOT is not imported.")
    print("Data paths, ROOT contents, and full training compatibility are not checked.")
    if errors:
        print("\nDependency check FAILED:")
        for error in errors:
            print(" -", error)
        return 1
    print("\nDependency check PASSED (imports and package constraints).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
