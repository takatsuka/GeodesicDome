#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_env.sh -- create (or repair) the Python environment for GeodesicDome (mt.geodesicdome).
#
# Run it once and you can immediately type, in the same terminal:
#
#   python examples/01_quickstart.py
#
# When it finishes, it opens a shell with the environment active (type `exit`
# to leave it), and it adds a small hook to ~/.zshrc (and ~/.bashrc) so that
# every new terminal activates the environment inside this project.
#
#   ./setup_env.sh                  # create or update the environment
#   ./setup_env.sh --check          # only verify that everything works
#   ./setup_env.sh --help           # all options
#   ./setup_env.sh --gpu none       # skip the GPU packages (torch, cupy)
#
# GPU packages are installed to match this machine: PyTorch for an NVIDIA
# (CUDA), Apple Silicon (MPS) or AMD (ROCm) GPU, and CuPy on NVIDIA.  Without a
# GPU nothing extra is installed and computations use every CPU core.
#
# Safe to run again at any time: an existing, healthy environment is reused
# and only missing packages are installed.  A broken one (for example after
# `brew upgrade python` removed the Python it was built from) is rebuilt.
#
# Works with the bash 3.2 that ships with macOS.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_NAME="GeodesicDome"
MIN_PY_MINOR=10                         # Python >= 3.10 (plane.py uses match/case)

# Use a project-local environment by default. Set MTGEODESICDOME_VENV or pass
# --venv to keep it outside the repository (recommended for cloud-synced folders).
VENV_DIR="${MTGEODESICDOME_VENV:-$REPO_DIR/.venv}"
PYTHON_BIN=""
EXTRAS="all"
RECREATE=0
# Editable installs use setuptools' "compat" mode: it puts each checkout's src/ folder on the
# path with a plain .pth file.  The default mode uses an import hook instead, which Python
# follows but PyCharm / VS Code cannot, so the editor reports e.g.
# "Cannot find reference 'geodesicdome' in 'mt'" even though the code runs.
EDITABLE_OPTS="--config-settings editable_mode=compat"
CHECK_ONLY=0
SHELL_HOOK=1                            # auto-activate in new terminals (--no-shell-hook to skip)
START_SHELL=1                           # open an activated shell at the end (--no-shell to skip)
REMOVE_HOOK=0
# GPU packages (step 4): auto = install what this machine's GPU can use (torch for CUDA / Apple
# Silicon / ROCm, plus CuPy on NVIDIA); torch or cupy = only that one; none = skip.
GPU_MODE="${MTGEODESIC_GPU:-}"
TORCH_INDEX="${MTGEODESIC_TORCH_INDEX:-}"   # override the PyTorch wheel index (e.g. cu128, rocm6.4 or a URL)
GPU_KIND="none"                             # detected below: cuda, mps, rocm or none
CUDA_VER=""                                 # CUDA version the NVIDIA driver supports, e.g. 12.8

HOOK_BEGIN="# >>> ${PROJECT_NAME} auto-activate (added by setup_env.sh) >>>"
HOOK_END="# <<< ${PROJECT_NAME} auto-activate <<<"

usage() {
  cat <<EOF
Usage: ./setup_env.sh [options]

Creates or updates the virtual environment for this project and installs the
package in editable mode with its dependencies.

Options:
  --python PATH       Python to build the environment with (must be >= 3.${MIN_PY_MINOR}).
                      Default: the newest python3.x >= 3.${MIN_PY_MINOR} found.
  --venv PATH         Environment directory (default: ${REPO_DIR}/.venv,
                      or \$MTGEODESICDOME_VENV if set).
  --minimal           Install only numpy (the core library).
  --no-legacy         Skip plotly and dash (only needed by the older viewers in
                      examples/legacy/).
  --gpu MODE          GPU packages: auto (default) installs what this machine's GPU
                      can use -- PyTorch for an NVIDIA (CUDA), Apple Silicon (MPS)
                      or AMD (ROCm) GPU, plus CuPy on NVIDIA; torch or cupy installs
                      only that one (torch also works without a GPU); none skips them.
                      Default: \$MTGEODESIC_GPU, else auto (none with --minimal).
  --no-gpu            Same as --gpu none.
  --torch-index IDX   PyTorch wheel index to use instead of the detected one, e.g.
                      cu128, cu126, rocm6.4, or a full URL (\$MTGEODESIC_TORCH_INDEX).
  --recreate          Delete the environment and build it from scratch.
  --no-shell          Do not open an activated shell at the end.
  --no-shell-hook     Do not add the auto-activation hook to ~/.zshrc / ~/.bashrc.
                      (By default it is added: new terminals activate the
                      environment inside this project and deactivate it outside.)
  --remove-shell-hook Remove that hook again.
  --check             Do not install anything; just verify the environment.
  -h, --help          Show this help.

In any terminal, you can also activate it yourself:
  source "${VENV_DIR}/bin/activate"
or call its Python directly from anywhere:
  "${VENV_DIR}/bin/python" path/to/script.py
EOF
}

# ------------------------------------------------------------------ helpers
say()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# prints "3.12.4" for a python executable, or nothing if it does not run
py_version() {
  "$1" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true
}

# true if the python executable is new enough
py_ok() {
  "$1" -c "import sys; sys.exit(0 if sys.version_info >= (3, ${MIN_PY_MINOR}) else 1)" >/dev/null 2>&1
}

# finds the newest suitable python3.x on PATH or in the usual Homebrew places
find_python() {
  local minor candidate path
  for minor in 14 13 12 11 10; do
    for candidate in "python3.${minor}" "/opt/homebrew/bin/python3.${minor}" "/usr/local/bin/python3.${minor}"; do
      path="$(command -v "$candidate" 2>/dev/null || true)"
      if [[ -n "$path" ]] && py_ok "$path"; then
        echo "$path"
        return 0
      fi
    done
  done
  for candidate in python3 /opt/homebrew/bin/python3 /usr/local/bin/python3 python; do
    path="$(command -v "$candidate" 2>/dev/null || true)"
    if [[ -n "$path" ]] && py_ok "$path"; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

# removes our block (between the markers) from a shell rc file
strip_hook() {
  local rc="$1"
  [[ -f "$rc" ]] || return 0
  grep -qF "$HOOK_BEGIN" "$rc" || return 0
  local tmp
  tmp="$(mktemp)"
  awk -v b="$HOOK_BEGIN" -v e="$HOOK_END" '
    $0 == b {skip = 1; next}
    $0 == e {skip = 0; next}
    !skip   {print}' "$rc" > "$tmp"
  cat "$tmp" > "$rc"          # keep the original file (and its permissions/links)
  rm -f "$tmp"
}

# writes the auto-activation hook into a shell rc file
install_hook() {
  local rc="$1" kind="$2"
  strip_hook "$rc"
  {
    echo "$HOOK_BEGIN"
    echo "_mtgeodesicdome_root=$(printf '%q' "$REPO_DIR")"
    echo "_mtgeodesicdome_venv=$(printf '%q' "$VENV_DIR")"
    cat <<'EOF'
_mtgeodesicdome_auto() {
  local here
  here="$(pwd -P 2>/dev/null || pwd)"
  case "$here/" in
    "$_mtgeodesicdome_root"/*)
      if [ "${VIRTUAL_ENV:-}" != "$_mtgeodesicdome_venv" ] && [ -f "$_mtgeodesicdome_venv/bin/activate" ]; then
        . "$_mtgeodesicdome_venv/bin/activate"
        _mtgeodesicdome_active=1
      fi ;;
    *)
      if [ -n "${_mtgeodesicdome_active:-}" ] && [ "${VIRTUAL_ENV:-}" = "$_mtgeodesicdome_venv" ]; then
        deactivate
      fi
      unset _mtgeodesicdome_active ;;
  esac
}
EOF
    if [[ "$kind" == zsh ]]; then
      echo 'autoload -Uz add-zsh-hook && add-zsh-hook chpwd _mtgeodesicdome_auto'
    else
      # shellcheck disable=SC2016  # written literally into ~/.bashrc
      echo 'case ";${PROMPT_COMMAND:-};" in *";_mtgeodesicdome_auto;"*) ;; *) PROMPT_COMMAND="_mtgeodesicdome_auto${PROMPT_COMMAND:+;$PROMPT_COMMAND}" ;; esac'
    fi
    echo '_mtgeodesicdome_auto'
    echo "$HOOK_END"
  } >> "$rc"
  echo "  added auto-activation to $rc"
}

# replaces this script with an interactive shell in which the environment is
# active.  The user's own rc file is read first, so their prompt, aliases and
# PATH set-up (Homebrew, conda, pyenv, ...) stay as usual; the environment is
# activated last so that its `python` wins.
start_activated_shell() {
  local shell_path shell_name rcdir
  shell_path="${SHELL:-/bin/zsh}"
  shell_name="$(basename "$shell_path")"
  rcdir="$VENV_DIR/.shell"
  mkdir -p "$rcdir"
  case "$shell_name" in
    zsh)
      # zsh reads $ZDOTDIR/.zshenv and $ZDOTDIR/.zshrc; point ZDOTDIR at small
      # files that load the user's own ones and then activate the environment.
      cat > "$rcdir/.zshenv" <<EOF
_mtgd_zd="\$ZDOTDIR"; ZDOTDIR="\${_MTGD_ORIG_ZDOTDIR:-\$HOME}"
[[ -f "\$ZDOTDIR/.zshenv" ]] && source "\$ZDOTDIR/.zshenv"
ZDOTDIR="\$_mtgd_zd"
EOF
      cat > "$rcdir/.zshrc" <<EOF
ZDOTDIR="\${_MTGD_ORIG_ZDOTDIR:-\$HOME}"; unset _mtgd_zd _MTGD_ORIG_ZDOTDIR
[[ -f "\$ZDOTDIR/.zshrc" ]] && source "\$ZDOTDIR/.zshrc"
source "$VENV_DIR/bin/activate"
EOF
      export _MTGD_ORIG_ZDOTDIR="${ZDOTDIR:-$HOME}"
      ZDOTDIR="$rcdir" exec "$shell_path" -i ;;
    bash)
      cat > "$rcdir/bashrc" <<EOF
if [ -f "\$HOME/.bashrc" ]; then . "\$HOME/.bashrc"; elif [ -f "\$HOME/.bash_profile" ]; then . "\$HOME/.bash_profile"; fi
. "$VENV_DIR/bin/activate"
EOF
      exec "$shell_path" --rcfile "$rcdir/bashrc" -i ;;
    fish)
      exec "$shell_path" -i -C "source '$VENV_DIR/bin/activate.fish'" ;;
    *)
      echo "(your shell, $shell_name, is not supported for auto-start; run:  source \"$VENV_DIR/bin/activate\")"
      return 0 ;;
  esac
}

# ------------------------------------------------------------------ GPU
# prints the CUDA version the NVIDIA driver supports (e.g. "12.8"), or nothing
nvidia_cuda_version() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  { nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1; } || true
}

# sets GPU_KIND (cuda, mps, rocm or none) and CUDA_VER
detect_gpu() {
  GPU_KIND="none"; CUDA_VER=""
  case "$(uname -s)" in
    Darwin)
      # Apple Silicon (also when this shell itself runs under Rosetta)
      if [[ "$(uname -m)" == arm64 ]] || [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == 1 ]]; then
        GPU_KIND="mps"
      fi ;;
    *)
      CUDA_VER="$(nvidia_cuda_version)"
      if [[ -n "$CUDA_VER" ]]; then
        GPU_KIND="cuda"
      elif command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
        GPU_KIND="rocm"
      fi ;;
  esac
}

# ver_ge 12.8 12.6  -> true
ver_ge() {
  local a1="${1%%.*}" a2="${1#*.}" b1="${2%%.*}" b2="${2#*.}"
  [[ "$a1" -gt "$b1" ]] || { [[ "$a1" -eq "$b1" ]] && [[ "$a2" -ge "$b2" ]]; }
}

# the PyTorch CUDA wheel index for the CUDA version the driver supports
torch_cuda_index() {
  local v="$1"
  if   ver_ge "$v" 13.0; then echo cu130
  elif ver_ge "$v" 12.8; then echo cu128
  elif ver_ge "$v" 12.6; then echo cu126
  elif ver_ge "$v" 12.4; then echo cu124
  elif ver_ge "$v" 11.8; then echo cu118
  fi
}

index_url() {
  case "$1" in
    http*) echo "$1" ;;
    *)     echo "https://download.pytorch.org/whl/$1" ;;
  esac
}

# true if the environment's torch imports and can use the GPU kind given (cuda, rocm, mps; none = imports)
torch_sees() {
  "$VENV_PY" - "$1" >/dev/null 2>&1 <<'PY'
import sys
import torch
kind = sys.argv[1]
if kind in ('cuda', 'rocm'):
    ok = torch.cuda.is_available() and (kind == 'cuda') == (torch.version.hip is None)
    ok = ok and float(torch.ones(4, device='cuda').sum()) == 4.0
elif kind == 'mps':
    ok = torch.backends.mps.is_available() and float(torch.ones(4, device='mps').sum()) == 4.0
else:
    ok = True
sys.exit(0 if ok else 1)
PY
}

cupy_works() {
  "$VENV_PY" -c 'import cupy; assert int((cupy.arange(4) ** 2).sum()) == 14' >/dev/null 2>&1
}

pkg_version() {
  "$VENV_PY" -c "import $1; print($1.__version__)" 2>/dev/null || echo "?"
}

install_torch() {
  local idx url
  if torch_sees "$GPU_KIND"; then
    if [[ "$GPU_KIND" == none ]]; then
      echo "      torch $(pkg_version torch) already installed"
    else
      echo "      torch $(pkg_version torch) already installed and uses the GPU ($GPU_KIND)"
    fi
    return 0
  fi
  case "$GPU_KIND" in
    mps)
      if [[ "$("$VENV_PY" -c 'import platform; print(platform.machine())')" != arm64 ]]; then
        warn "this environment's Python is an Intel (x86_64) build running under Rosetta; PyTorch"
        warn "cannot use the Apple GPU from it.  Rebuild with an Apple Silicon Python, e.g."
        warn "  ./setup_env.sh --recreate --python /opt/homebrew/bin/python3.13"
        return 1
      fi
      echo "      installing torch (Apple Silicon GPU through MPS)"
      "$VENV_PY" -m pip install --quiet --upgrade torch ;;
    cuda)
      idx="${TORCH_INDEX:-$(torch_cuda_index "$CUDA_VER")}"
      if [[ -z "$idx" ]]; then
        warn "the NVIDIA driver supports CUDA $CUDA_VER, older than any current PyTorch build (11.8+); update the driver"
        return 1
      fi
      url="$(index_url "$idx")"
      echo "      installing torch for CUDA (the driver supports CUDA $CUDA_VER) from $url"
      # --force-reinstall replaces a CPU-only build; if the download fails, nothing is changed
      "$VENV_PY" -m pip install --quiet --upgrade --force-reinstall torch --index-url "$url" ;;
    rocm)
      for idx in ${TORCH_INDEX:-rocm7.0 rocm6.4 rocm6.3}; do
        url="$(index_url "$idx")"
        echo "      installing torch for an AMD GPU (ROCm) from $url"
        if "$VENV_PY" -m pip install --quiet --upgrade --force-reinstall torch --index-url "$url" 2>/dev/null; then
          break
        fi
      done ;;
    *)
      echo "      installing torch (no GPU found: CPU build)"
      "$VENV_PY" -m pip install --quiet --upgrade torch ;;
  esac
  torch_sees "$GPU_KIND"
}

install_cupy() {
  local major="${CUDA_VER%%.*}" pkg other
  case "$major" in
    11|12|13) pkg="cupy-cuda${major}x" ;;
    *) warn "no CuPy build for CUDA $CUDA_VER"; return 1 ;;
  esac
  if cupy_works; then
    echo "      cupy $(pkg_version cupy) already installed (uses the GPU)"
    return 0
  fi
  # CuPy builds for other CUDA versions would shadow this one
  for other in cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x; do
    [[ "$other" == "$pkg" ]] || "$VENV_PY" -m pip uninstall --quiet -y "$other" >/dev/null 2>&1 || true
  done
  echo "      installing $pkg"
  "$VENV_PY" -m pip install --quiet --upgrade "$pkg" || return 1
  cupy_works && return 0
  echo "      $pkg needs the CUDA runtime libraries; installing them as Python packages (${pkg}[ctk])"
  "$VENV_PY" -m pip install --quiet --upgrade "${pkg}[ctk]" || return 1
  cupy_works
}

# step 4: the packages that let mt.geodesicdome.backend use the GPU
install_gpu_packages() {
  local failed=0
  if [[ "$GPU_MODE" == none ]]; then
    echo "      skipped (--gpu none); computations will use every CPU core"
    return 0
  fi
  case "$GPU_KIND" in
    cuda) echo "      found an NVIDIA GPU (driver supports CUDA $CUDA_VER)" ;;
    mps)  echo "      found an Apple Silicon GPU (Metal / MPS)" ;;
    rocm) echo "      found an AMD GPU (ROCm)" ;;
    none)
      if [[ "$GPU_MODE" == auto ]]; then
        echo "      no GPU found (no nvidia-smi, ROCm or Apple Silicon); computations will use every"
        echo "      CPU core, which needs nothing extra"
        return 0
      fi ;;
  esac
  if [[ "$GPU_MODE" == auto ]] || [[ "$GPU_MODE" == torch ]]; then
    install_torch || { warn "PyTorch could not use the GPU (see above)"; failed=1; }
  fi
  if [[ "$GPU_MODE" == auto ]] || [[ "$GPU_MODE" == cupy ]]; then
    if [[ "$GPU_KIND" == cuda ]]; then
      install_cupy || { warn "CuPy could not use the GPU (see above)"; failed=1; }
    elif [[ "$GPU_MODE" == cupy ]]; then
      warn "CuPy needs an NVIDIA GPU; none was found"; failed=1
    fi
  fi
  return "$failed"
}

# ------------------------------------------------------------ arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)            [[ $# -ge 2 ]] || die "--python needs a value"; PYTHON_BIN="$2"; shift 2 ;;
    --venv)              [[ $# -ge 2 ]] || die "--venv needs a value"; VENV_DIR="$2"; shift 2 ;;
    --minimal)           EXTRAS=""; shift ;;
    --no-legacy)         EXTRAS="interactive,examples,notebooks,dev"; shift ;;
    --with-legacy-viewers) EXTRAS="all"; shift ;;          # accepted for compatibility (now the default)
    --gpu)               [[ $# -ge 2 ]] || die "--gpu needs a value"; GPU_MODE="$2"; shift 2 ;;
    --no-gpu)            GPU_MODE="none"; shift ;;
    --torch-index)       [[ $# -ge 2 ]] || die "--torch-index needs a value"; TORCH_INDEX="$2"; shift 2 ;;
    --recreate)          RECREATE=1; shift ;;
    --shell-hook)        SHELL_HOOK=1; shift ;;         # accepted for compatibility (now the default)
    --no-shell-hook)     SHELL_HOOK=0; shift ;;
    --no-shell)          START_SHELL=0; shift ;;
    --remove-shell-hook) REMOVE_HOOK=1; shift ;;
    --check)             CHECK_ONLY=1; START_SHELL=0; SHELL_HOOK=0; shift ;;
    -h|--help)           usage; exit 0 ;;
    *)                   usage >&2; die "unknown option: $1" ;;
  esac
done

# make the venv path absolute (relative paths are relative to where you run the script)
case "$VENV_DIR" in
  /*) ;;
  "~"*) VENV_DIR="$HOME${VENV_DIR#\~}" ;;
  *)  VENV_DIR="$PWD/$VENV_DIR" ;;
esac
VENV_PY="$VENV_DIR/bin/python"

if [[ -z "$GPU_MODE" ]]; then
  if [[ -z "$EXTRAS" ]]; then GPU_MODE="none"; else GPU_MODE="auto"; fi   # --minimal: core only
fi
case "$GPU_MODE" in
  auto|torch|cupy|none) ;;
  *) die "--gpu must be auto, torch, cupy or none (not '$GPU_MODE')" ;;
esac
detect_gpu

if [[ "$REMOVE_HOOK" -eq 1 ]]; then
  for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do strip_hook "$rc"; done
  say "Removed the ${PROJECT_NAME} auto-activation hook (open a new terminal)."
  exit 0
fi

case "$VENV_DIR" in
  *"/Library/CloudStorage/"*|*"/Google Drive"*|*"/My Drive/"*|*"/Dropbox/"*|*"/OneDrive"*)
    warn "the environment is inside a cloud-synced folder ($VENV_DIR)."
    warn "this is known to cause slow or failing imports; ~/.venvs/${PROJECT_NAME} is recommended." ;;
esac

# ------------------------------------------------ 1. the environment
if [[ "$CHECK_ONLY" -eq 0 ]]; then
  # choose and validate the Python first, so nothing is deleted if it is unusable
  venv_healthy=0
  if [[ -x "$VENV_PY" ]] && py_ok "$VENV_PY"; then venv_healthy=1; fi
  if [[ -n "$PYTHON_BIN" ]]; then
    command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || die "python not found: $PYTHON_BIN"
    if ! py_ok "$PYTHON_BIN"; then
      found="$(py_version "$PYTHON_BIN")"
      die "$PYTHON_BIN is ${found:+Python $found}${found:-not a working Python}; 3.${MIN_PY_MINOR} or newer is required"
    fi
  elif [[ "$RECREATE" -eq 1 ]] || [[ "$venv_healthy" -eq 0 ]]; then
    PYTHON_BIN="$(find_python)" || die "no Python >= 3.${MIN_PY_MINOR} found. Install one, e.g.  brew install python@3.13"
  fi

  # reuse an existing, working environment unless asked to rebuild it
  if [[ -d "$VENV_DIR" ]]; then
    if [[ "$RECREATE" -eq 1 ]]; then
      say "[1/5] Removing $VENV_DIR (--recreate)"
      rm -rf "$VENV_DIR"
    elif [[ "$venv_healthy" -eq 1 ]]; then
      if [[ -n "$PYTHON_BIN" ]] && [[ "$(py_version "$PYTHON_BIN")" != "$(py_version "$VENV_PY")" ]]; then
        say "[1/5] Existing environment uses Python $(py_version "$VENV_PY"); rebuilding with $PYTHON_BIN"
        rm -rf "$VENV_DIR"
      fi
    else
      say "[1/5] Existing environment at $VENV_DIR is broken or too old; rebuilding it"
      rm -rf "$VENV_DIR"
    fi
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    say "[1/5] Creating environment with $PYTHON_BIN (Python $(py_version "$PYTHON_BIN"))"
    echo "      at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  else
    say "[1/5] Using existing environment $VENV_DIR (Python $(py_version "$VENV_PY"))"
  fi

  # ---------------------------------------------- 2. packages
  say "[2/5] Updating pip"
  "$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel

  say "[3/5] Installing ${PROJECT_NAME} (editable) and its dependencies"
  if [[ -n "$EXTRAS" ]]; then
    "$VENV_PY" -m pip install --quiet $EDITABLE_OPTS -e "${REPO_DIR}[${EXTRAS}]"
  else
    "$VENV_PY" -m pip install --quiet $EDITABLE_OPTS -e "${REPO_DIR}"
  fi
  # ---------------------------------------------- 3. GPU packages
  say "[4/5] GPU packages (--gpu $GPU_MODE)"
  install_gpu_packages || true            # any problem is reported by the verification below
else
  [[ -x "$VENV_PY" ]] || die "no environment at $VENV_DIR -- run ./setup_env.sh first"
fi

# -------------------------------------------------- 5. verification
say "[5/5] Verifying"
# run from a neutral folder, to prove the package is importable from anywhere
(cd / && MTG_GPU_KIND="$GPU_KIND" MTG_GPU_MODE="$GPU_MODE" "$VENV_PY" - "$REPO_DIR" "$EXTRAS" "$CHECK_ONLY" <<'PY'
import importlib, os, sys

repo, extras, check_only = os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3] == '1'
required = ['numpy']
if extras:
    required += ['matplotlib', 'PIL', 'pytest']
optional = ['plotly', 'dash']                      # only for the old viewers in examples/legacy/
if extras == 'all' and not check_only:
    required += optional                           # they were just installed, so they must work

problems = []
print(f'      python      {sys.version.split()[0]}  ({sys.executable})')
for name in required:
    try:
        module = importlib.import_module(name)
        print(f'      {name:<11} {getattr(module, "__version__", "ok")}')
    except Exception as exc:                       # noqa: BLE001
        problems.append(f'{name}: {exc}')
if check_only:
    for name in optional:
        try:
            module = importlib.import_module(name)
            print(f'      {name:<11} {getattr(module, "__version__", "ok")}')
        except Exception:                          # noqa: BLE001
            print(f'      {name:<11} not installed (optional: only examples/legacy/ needs it)')

try:
    import mt.geodesicdome
    from mt.geodesicdome.grid.geodesicdome import GeodesicDome
    location = os.path.realpath(list(mt.geodesicdome.__path__)[0])
    if not location.startswith(repo):
        problems.append(f'mt.geodesicdome is imported from {location}, not from this repository')
    dome = GeodesicDome(4)
    assert len(dome.get_all_triangles()) == 3 * 320
    print(f'      mt.geodesicdome {location}')
except Exception as exc:                           # noqa: BLE001
    problems.append(f'mt.geodesicdome: {exc}')

if extras:
    try:
        import matplotlib
        from mt.geodesicdome.interactive import ProjectionViewer  # noqa: F401
        backend = matplotlib.get_backend()
        print(f'      mpl backend {backend}')
        if backend.lower() in ('agg', 'pdf', 'svg', 'ps', 'template') and sys.platform == 'darwin':
            problems.append(f'matplotlib backend is {backend!r}; interactive windows will not open '
                            '(check MPLBACKEND or your matplotlibrc)')
    except Exception as exc:                       # noqa: BLE001
        problems.append(f'interactive viewer: {exc}')

# the compute backend: GPU if the machine has one, else every CPU core (mt.geodesicdome.backend)
gpu_kind, gpu_mode = os.environ.get('MTG_GPU_KIND', 'none'), os.environ.get('MTG_GPU_MODE', 'auto')
for name in ('torch', 'cupy'):
    try:
        module = importlib.import_module(name)
        print(f'      {name:<11} {module.__version__}')
    except Exception:                              # noqa: BLE001
        print(f'      {name:<11} not installed')
try:
    import numpy as np
    from mt.geodesicdome import backend as bk
    try:
        b = bk.get_backend()
        a = b.asarray(np.arange(6.0).reshape(2, 3))
        assert abs(float(b.to_numpy(b.sum(a @ a.T))) - 83.0) < 1e-3
        print(f'      compute     {b}')
    except Exception as exc:                       # noqa: BLE001
        problems.append(f'compute backend: {exc}  (MTGEODESIC_BACKEND={os.environ.get("MTGEODESIC_BACKEND", "")!r})')
        b = None
    if b is not None and gpu_kind != 'none' and gpu_mode != 'none' and not b.is_gpu:
        hint = ('run ./setup_env.sh to install the GPU packages' if check_only
                else 'the GPU packages did not install correctly (see the warnings above)')
        problems.append(f'this machine has a GPU ({gpu_kind}) but computations would run on {b}: {hint}; '
                        'or use --gpu none to go without')
    if gpu_kind == 'cuda' and gpu_mode in ('auto', 'cupy'):
        try:
            import cupy
            assert int((cupy.arange(4) ** 2).sum()) == 14
            print('      cupy GPU    ok')
        except Exception as exc:                   # noqa: BLE001
            problems.append(f'cupy on the NVIDIA GPU: {exc}')
except ImportError:
    print('      compute     this geodesicdomes has no GPU backend (needs geodesicdomes >= 1.3.0)')

if problems:
    print('\nPROBLEMS:')
    for p in problems:
        print('  -', p)
    sys.exit(1)
PY
) || die "verification failed (see above). Try:  ./setup_env.sh --recreate"

# ---------------------------------------------------- 6. shell hook
if [[ "$SHELL_HOOK" -eq 1 ]]; then
  say "Installing the auto-activation hook"
  install_hook "$HOME/.zshrc" zsh
  if [[ -f "$HOME/.bashrc" ]]; then install_hook "$HOME/.bashrc" bash; fi
fi

# ----------------------------------------------------------- summary
echo
say "Environment is ready."
hook_present=0
if grep -qsF "$HOOK_BEGIN" "$HOME/.zshrc" || grep -qsF "$HOOK_BEGIN" "$HOME/.bashrc"; then hook_present=1; fi

if [[ "${VIRTUAL_ENV:-}" == "$VENV_DIR" ]]; then
  # this terminal already uses the environment
  cat <<EOF
This terminal already uses it, so you can run, e.g.
  python examples/01_quickstart.py
EOF
elif [[ "$START_SHELL" -eq 1 ]] && [[ -t 0 ]] && [[ -t 1 ]]; then
  cat <<EOF
Opening a shell with the environment active, so you can run straight away, e.g.
  python examples/01_quickstart.py
  python examples/09_interactive_projection.py
Type 'exit' to return to your previous shell.
EOF
  [[ "$hook_present" -eq 1 ]] && echo "New terminals activate it automatically whenever you are inside $REPO_DIR"
  echo
  start_activated_shell
else
  cat <<EOF
To use it in this terminal:
  source "$VENV_DIR/bin/activate"
Then any script runs, e.g.
  python examples/01_quickstart.py
EOF
  [[ "$hook_present" -eq 1 ]] && echo "New terminals activate it automatically whenever you are inside $REPO_DIR"
fi
