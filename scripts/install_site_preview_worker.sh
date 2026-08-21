#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this installer as root." >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker_root="/opt/aiv-preview"
worker_home="/var/lib/aiv-preview"

if ! id aiv-preview >/dev/null 2>&1; then
  useradd \
    --system \
    --home-dir "${worker_home}" \
    --create-home \
    --shell /usr/sbin/nologin \
    aiv-preview
fi

install -d -m 0755 "${worker_root}"
install -d -o aiv-preview -g aiv-preview -m 0750 "${worker_home}"
python3 -m venv "${worker_root}/venv"
"${worker_root}/venv/bin/pip" install --disable-pip-version-check "playwright>=1.52.0"
install -m 0644 \
  "${project_root}/app/services/site_preview_worker.py" \
  "${worker_root}/site_preview_worker.py"
install -m 0755 \
  "${project_root}/scripts/aiv-site-preview" \
  /usr/local/bin/aiv-site-preview

PLAYWRIGHT_BROWSERS_PATH="${worker_home}/browsers" \
  "${worker_root}/venv/bin/python" -m playwright install --with-deps chromium
chown -R aiv-preview:aiv-preview "${worker_home}"

echo "Installed isolated site-preview worker at /usr/local/bin/aiv-site-preview"
