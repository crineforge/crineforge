import subprocess
import glob
files = glob.glob("dist/*")
result = subprocess.run(
    ["python", "-m", "twine", "upload", "--verbose", "--username", "__token__", "--password", "pypi-AgEIcHlwaS5vcmcCJGUxNzc5YzcyLTNiNzgtNDI5OC1hMDJiLTkzNGE4YWEyNTViMAACKlszLCI3ZjBiN2E4Mi1iYjNiLTQzYjAtOTgyYS0wZTEyY2JjNzFhYzciXQAABiDo2heTuNoH96X1uz9Te_6RxdEP_XEBUkkXjvRJHA-4VA"] + files,
    capture_output=True, text=True, encoding="utf-8", errors="ignore"
)
with open("upload_clean.log", "w", encoding="utf-8") as f:
    f.write("--- STDOUT ---\n")
    f.write(result.stdout)
    f.write("\n--- STDERR ---\n")
    f.write(result.stderr)

