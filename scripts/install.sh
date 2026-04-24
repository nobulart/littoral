#!/usr/bin/env bash
set -euo pipefail

LITTORAL_REPO_URL="${LITTORAL_REPO_URL:-https://github.com/nobulart/littoral.git}"
LITTORAL_INSTALL_DIR="${LITTORAL_INSTALL_DIR:-$HOME/.local/share/littoral}"
LITTORAL_VENV_DIR="${LITTORAL_VENV_DIR:-.venv}"
LITTORAL_MODELS="${LITTORAL_MODELS:-mistral-small:24b qwen2.5vl:7b glm-ocr:latest}"
LITTORAL_ASSUME_YES="${LITTORAL_ASSUME_YES:-0}"
LITTORAL_SKIP_MODELS="${LITTORAL_SKIP_MODELS:-0}"
LITTORAL_SKIP_OLLAMA="${LITTORAL_SKIP_OLLAMA:-0}"

REQUIRED_PYTHON_MIN="3.10"
REQUIRED_PYTHON_MAX_EXCLUSIVE="3.14"
ESTIMATED_PYTHON_GB=22
ESTIMATED_SYSTEM_GB=2
ESTIMATED_OLLAMA_GB=1

usage() {
  cat <<'EOF'
Install LITTORAL and its local runtime dependencies.

Usage:
  curl -fsSL https://raw.githubusercontent.com/nobulart/littoral/main/scripts/install.sh | bash
  bash scripts/install.sh [--yes] [--install-dir PATH] [--models "model:tag ..."]

Options:
  -y, --yes          Run non-interactively.
  --install-dir DIR  Clone/use LITTORAL at DIR when not run from a checkout.
  --models LIST      Space-separated Ollama models to pull.
  --skip-models      Install Ollama but do not pull models.
  --skip-ollama      Do not install Ollama or pull models.
  -h, --help         Show this help.
EOF
}

log() {
  printf '>>> %s\n' "$*" >&2
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -y|--yes)
      LITTORAL_ASSUME_YES=1
      ;;
    --install-dir)
      shift
      [ "$#" -gt 0 ] || die "--install-dir requires a path"
      LITTORAL_INSTALL_DIR="$1"
      ;;
    --models)
      shift
      [ "$#" -gt 0 ] || die "--models requires a quoted model list"
      LITTORAL_MODELS="$1"
      ;;
    --skip-models)
      LITTORAL_SKIP_MODELS=1
      ;;
    --skip-ollama)
      LITTORAL_SKIP_OLLAMA=1
      LITTORAL_SKIP_MODELS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
  shift
done

detect_os() {
  case "$(uname -s)" in
    Darwin) printf 'macos' ;;
    Linux) printf 'linux' ;;
    *) die "unsupported operating system: $(uname -s). LITTORAL installer supports macOS and Linux." ;;
  esac
}

detect_arch() {
  case "$(uname -m)" in
    arm64|aarch64) printf 'arm64' ;;
    x86_64|amd64) printf 'amd64' ;;
    *) die "unsupported CPU architecture: $(uname -m)" ;;
  esac
}

version_check_python() {
  local python_bin="$1"
  "$python_bin" - "$REQUIRED_PYTHON_MIN" "$REQUIRED_PYTHON_MAX_EXCLUSIVE" <<'PY'
import sys
minimum = tuple(map(int, sys.argv[1].split(".")))
maximum = tuple(map(int, sys.argv[2].split(".")))
current = sys.version_info[:2]
sys.exit(0 if minimum <= current < maximum else 1)
PY
}

python_version_text() {
  "$1" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
}

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if have "$candidate" && version_check_python "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  return 1
}

disk_available_gb() {
  local target="$1"
  while [ ! -e "$target" ] && [ "$target" != "/" ]; do
    target="$(dirname "$target")"
  done
  if df -Pk "$target" >/dev/null 2>&1; then
    df -Pk "$target" | awk 'NR==2 { printf "%.0f", $4 / 1024 / 1024 }'
  else
    printf '0'
  fi
}

memory_gb() {
  local os_name="$1"
  if [ "$os_name" = "macos" ]; then
    sysctl -n hw.memsize 2>/dev/null | awk '{ printf "%.0f", $1 / 1024 / 1024 / 1024 }'
  elif have free; then
    free -g | awk '/^Mem:/ { print $2 }'
  else
    printf '0'
  fi
}

model_size_gb() {
  case "$1" in
    mistral-small:24b|mistral-small:latest) printf '14' ;;
    qwen2.5vl:7b|qwen2.5vl:latest) printf '6' ;;
    glm-ocr:latest|glm-ocr) printf '3' ;;
    llama3.2-vision:11b) printf '8' ;;
    qwen3:8b) printf '6' ;;
    qwen2.5:7b) printf '5' ;;
    *) printf '8' ;;
  esac
}

estimated_model_gb() {
  if [ "$LITTORAL_SKIP_MODELS" = "1" ]; then
    printf '0'
    return
  fi
  local total=0
  local model
  for model in $LITTORAL_MODELS; do
    total=$((total + $(model_size_gb "$model")))
  done
  printf '%s' "$total"
}

confirm() {
  [ "$LITTORAL_ASSUME_YES" = "1" ] && return 0
  [ -t 0 ] || [ -r /dev/tty ] || die "non-interactive install requires --yes"
  printf 'Proceed with installation? [y/N] '
  local reply
  if [ -r /dev/tty ]; then
    read -r reply </dev/tty
  else
    read -r reply
  fi
  case "$reply" in
    y|Y|yes|YES) ;;
    *) die "installation cancelled" ;;
  esac
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_macos_packages() {
  local packages=(poppler tesseract git curl)
  if ! have brew; then
    log "Homebrew is not installed; installing Homebrew so system packages can be managed."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -x /opt/homebrew/bin/brew ]; then
      eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -x /usr/local/bin/brew ]; then
      eval "$(/usr/local/bin/brew shellenv)"
    fi
  fi
  have brew || die "Homebrew installation did not make brew available on PATH"
  log "Installing system packages with Homebrew: ${packages[*]}"
  brew install "${packages[@]}"
}

install_linux_packages() {
  if have apt-get; then
    log "Installing system packages with apt-get"
    run_sudo apt-get update
    run_sudo apt-get install -y git curl ca-certificates python3-venv python3-pip poppler-utils tesseract-ocr
  elif have dnf; then
    log "Installing system packages with dnf"
    run_sudo dnf install -y git curl ca-certificates python3 python3-pip poppler-utils tesseract
  elif have yum; then
    log "Installing system packages with yum"
    run_sudo yum install -y git curl ca-certificates python3 python3-pip poppler-utils tesseract
  elif have pacman; then
    log "Installing system packages with pacman"
    run_sudo pacman -Sy --needed --noconfirm git curl ca-certificates python python-pip poppler tesseract
  else
    die "unsupported Linux package manager. Install git, curl, Python 3.10-3.13, Poppler, and Tesseract, then rerun."
  fi
}

ensure_repo() {
  if [ -f "requirements.txt" ] && [ -f "run_pipeline.py" ] && [ -d "src" ]; then
    pwd
    return
  fi

  if [ -d "$LITTORAL_INSTALL_DIR/.git" ]; then
    log "Updating existing checkout at $LITTORAL_INSTALL_DIR"
    git -C "$LITTORAL_INSTALL_DIR" pull --ff-only
  elif [ -d "$LITTORAL_INSTALL_DIR" ] && [ -n "$(find "$LITTORAL_INSTALL_DIR" -mindepth 1 -maxdepth 1 2>/dev/null)" ]; then
    die "$LITTORAL_INSTALL_DIR exists and is not an empty git checkout"
  else
    log "Cloning LITTORAL into $LITTORAL_INSTALL_DIR"
    mkdir -p "$(dirname "$LITTORAL_INSTALL_DIR")"
    git clone "$LITTORAL_REPO_URL" "$LITTORAL_INSTALL_DIR"
  fi
  printf '%s' "$LITTORAL_INSTALL_DIR"
}

ensure_ollama() {
  [ "$LITTORAL_SKIP_OLLAMA" = "1" ] && return
  if have ollama; then
    log "Ollama already installed: $(ollama --version 2>/dev/null || printf 'version unavailable')"
  else
    log "Installing Ollama from https://ollama.com/install.sh"
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  if ! curl -fsS --max-time 5 http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "Starting Ollama service"
    if [ "$(detect_os)" = "macos" ] && have open; then
      open -a Ollama --args hidden >/dev/null 2>&1 || true
      sleep 5
    else
      nohup ollama serve >/tmp/littoral-ollama.log 2>&1 &
      sleep 5
    fi
  fi
}

pull_models() {
  [ "$LITTORAL_SKIP_MODELS" = "1" ] && return
  local model
  for model in $LITTORAL_MODELS; do
    log "Pulling Ollama model: $model"
    ollama pull "$model"
  done
}

install_python_runtime() {
  local repo_dir="$1"
  local python_bin="$2"
  cd "$repo_dir"
  log "Creating Python virtual environment at $repo_dir/$LITTORAL_VENV_DIR"
  "$python_bin" -m venv "$LITTORAL_VENV_DIR"
  "$LITTORAL_VENV_DIR/bin/python" -m pip install --upgrade pip wheel
  "$LITTORAL_VENV_DIR/bin/python" -m pip install -r requirements.txt
}

verify_commands() {
  local repo_dir="$1"
  cd "$repo_dir"
  log "Verifying installed commands"
  "$LITTORAL_VENV_DIR/bin/python" run_pipeline.py --help >/dev/null
  "$LITTORAL_VENV_DIR/bin/mineru" --help >/dev/null || warn "mineru CLI did not respond to --help"
  have pdfinfo || warn "pdfinfo was not found on PATH"
  have pdftotext || warn "pdftotext was not found on PATH"
  have pdftoppm || warn "pdftoppm was not found on PATH"
  have tesseract || warn "tesseract was not found on PATH"
}

main() {
  local os_name arch python_bin mem_gb model_gb total_gb avail_gb repo_dir estimate_dir
  os_name="$(detect_os)"
  arch="$(detect_arch)"
  mem_gb="$(memory_gb "$os_name")"
  model_gb="$(estimated_model_gb)"
  total_gb=$((ESTIMATED_PYTHON_GB + ESTIMATED_SYSTEM_GB + ESTIMATED_OLLAMA_GB + model_gb))
  if [ -f "requirements.txt" ] && [ -f "run_pipeline.py" ] && [ -d "src" ]; then
    estimate_dir="$(pwd)"
  else
    estimate_dir="$LITTORAL_INSTALL_DIR"
  fi
  avail_gb="$(disk_available_gb "$estimate_dir")"

  log "LITTORAL installer"
  printf 'System: %s/%s\n' "$os_name" "$arch"
  [ "$mem_gb" = "0" ] || printf 'Memory: %s GB detected\n' "$mem_gb"
  printf 'Install directory: %s\n' "$estimate_dir"
  printf 'Estimated disk required: ~%s GB (%s GB Python/MinerU, %s GB system tools, %s GB Ollama app, %s GB models)\n' \
    "$total_gb" "$ESTIMATED_PYTHON_GB" "$ESTIMATED_SYSTEM_GB" "$ESTIMATED_OLLAMA_GB" "$model_gb"
  printf 'Available disk near install directory: ~%s GB\n' "$avail_gb"
  [ "$LITTORAL_SKIP_MODELS" = "1" ] || printf 'Ollama models: %s\n' "$LITTORAL_MODELS"

  if [ "$os_name" = "macos" ]; then
    local mac_major
    mac_major="$(sw_vers -productVersion | awk -F. '{ print $1 }')"
    [ "$mac_major" -ge 14 ] || warn "Ollama recommends macOS Sonoma 14 or newer."
  fi
  [ "$mem_gb" = "0" ] || [ "$mem_gb" -ge 16 ] || warn "MinerU recommends at least 16 GB RAM; 32 GB or more is better."
  [ "$avail_gb" -ge "$total_gb" ] || warn "Available disk appears below the estimate. Model pulls or MinerU install may fail."

  confirm

  if [ "$os_name" = "macos" ]; then
    install_macos_packages
  else
    install_linux_packages
  fi

  python_bin="$(find_python)" || die "Python $REQUIRED_PYTHON_MIN to < $REQUIRED_PYTHON_MAX_EXCLUSIVE is required"
  log "Using Python $(python_version_text "$python_bin") at $(command -v "$python_bin")"

  repo_dir="$(ensure_repo)"
  install_python_runtime "$repo_dir" "$python_bin"
  ensure_ollama
  pull_models
  verify_commands "$repo_dir"

  log "Installation complete"
  printf '\nNext steps:\n'
  printf '  cd %s\n' "$repo_dir"
  printf '  source %s/bin/activate\n' "$LITTORAL_VENV_DIR"
  printf '  python run_pipeline.py --fast-test\n'
}

main "$@"
