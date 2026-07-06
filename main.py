import os
import warnings

# Suppress multiprocessing resource_tracker shutdown warnings.
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*leaked semaphore.*')

# Also propagate warning suppression to multiprocessing child processes (resource_tracker runs in a
# separate Python process, so in-process filterwarnings() may not affect it).
_pw = os.environ.get("PYTHONWARNINGS", "")
_extra = "ignore::UserWarning:multiprocessing.resource_tracker"
if _extra not in _pw:
    os.environ["PYTHONWARNINGS"] = ",".join([p for p in [_pw, _extra] if p])

from gui.launcher import launcher

if __name__ == "__main__":
    launcher()