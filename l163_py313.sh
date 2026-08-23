#!/usr/bin/env bash
# L163a -- get a real Python 3.13 so the cp313 scipy wheel can actually be
# imported and tested. The grader states Python 3.13.14; this box has 3.10
# (conda) and 3.14 (WSL), and a cp313 wheel imports on neither.
set -u
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true
echo "pyenv: $(pyenv --version 2>&1)"
if pyenv versions --bare 2>/dev/null | grep -q '^3\.13\.'; then
  echo "already have: $(pyenv versions --bare | grep '^3\.13\.' | head -1)"
else
  echo "installing 3.13.14 (builds from source, several minutes)..."
  pyenv install -s 3.13.14 2>&1 | tail -5
fi
P="$PYENV_ROOT/versions/3.13.14/bin/python3.13"
[ -x "$P" ] && echo "OK: $($P --version)" || echo "FAILED to build 3.13"
