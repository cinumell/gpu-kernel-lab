from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def run(*command: str) -> None:
    if not shutil.which(command[0]):
        print(f"\n$ {' '.join(command)}\n<not found>")
        return
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    print(f"\n$ {' '.join(command)}")
    print((result.stdout or result.stderr).strip())


print(f"python={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
run("nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version,memory.total,power.limit", "--format=csv")
run("nvcc", "--version")
run(sys.executable, "-m", "pip", "freeze")

