"""Run scripts/psi0/encode_sonic_tokens.py on a machine with no Isaac Lab (e.g. Koa).

The script needs two leaf modules -- fiatlux_task.groot and fiatlux_task.psi0 -- and its own
docstring says it needs no simulator. But importing either one executes the package's
__init__.py, which does `from .tasks import *` and pulls in Isaac Lab.

Pre-seeding sys.modules with a bare `fiatlux_task` whose __path__ points at the real source
directory lets both submodules (and psi0.py's relative `from .groot import ...`) resolve to the
real files while __init__.py never runs. Nothing in the repo is modified.

    python run_encode.py <fiatlux-repo-root> [args for encode_sonic_tokens.py]
"""

import pathlib
import runpy
import sys
import types

repo = pathlib.Path(sys.argv.pop(1)).resolve()
source = repo / "source" / "fiatlux_task" / "fiatlux_task"
if not (source / "groot.py").is_file():
    sys.exit(f"FATAL: {source}/groot.py not found -- is {repo} a fiatlux checkout?")

pkg = types.ModuleType("fiatlux_task")
pkg.__path__ = [str(source)]
sys.modules["fiatlux_task"] = pkg

script = str(repo / "scripts" / "psi0" / "encode_sonic_tokens.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
