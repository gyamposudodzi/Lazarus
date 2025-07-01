import sys
import os

print("[DEBUG] Python path:", sys.executable)
print("[DEBUG] Python version:", sys.version)
print("[DEBUG] Environment variables:")
print(os.environ.get("VIRTUAL_ENV", "No virtual environment active"))
