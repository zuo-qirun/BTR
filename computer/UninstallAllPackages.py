def remove_all_global_pip_packages():
    import subprocess
    import sys

    pip_cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    output = subprocess.check_output(
        pip_cmd + ["list", "--format=freeze"], text=True
    ).strip()
    if not output:
        return
    pkgs = [line.split("==", 1)[0] for line in output.splitlines() if line]
    if not pkgs:
        return
    # Keep pip installed; batch uninstalls for speed.
    pkgs = [p for p in pkgs if p.lower() != "pip"]
    if not pkgs:
        return
    for i in range(0, len(pkgs), 100):
        subprocess.check_call(pip_cmd + ["uninstall", "-y", *pkgs[i : i + 100]])


if __name__ == "__main__":
    remove_all_global_pip_packages()
