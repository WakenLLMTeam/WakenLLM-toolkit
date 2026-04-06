#!/usr/bin/env bash
# 将本地 citation 相关目录同步到服务器 deerflow-2-zipeng-revise，覆盖同名文件夹。
# 用法（在 macOS/Linux 上）:
#   chmod +x scripts/sync_citation_dirs_to_zipeng.sh
#   ./scripts/sync_citation_dirs_to_zipeng.sh
#
# 可选环境变量:
#   REMOTE_HOST=root@47.100.52.201
#   REMOTE_DIR=/root/deerflow-2-zipeng-revise

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-root@47.100.52.201}"
REMOTE_DIR="${REMOTE_DIR:-/root/deerflow-2-zipeng-revise}"

echo "ROOT=$ROOT"
echo "REMOTE=${REMOTE_HOST}:${REMOTE_DIR}"

PACK="$(mktemp -d)/citation-pack"
rm -rf "$PACK"
mkdir -p "$PACK/backend/packages/harness/deerflow/utils" \
         "$PACK/backend/packages/harness/deerflow/agents/middlewares" \
         "$PACK/middlewares" \
         "$PACK/utils"

# 运行时引用：deerflow.utils / deerflow.agents.middlewares
cp -a "$ROOT/backend/packages/harness/deerflow/utils/." "$PACK/backend/packages/harness/deerflow/utils/"
cp -a "$ROOT/backend/packages/harness/deerflow/agents/middlewares/." "$PACK/backend/packages/harness/deerflow/agents/middlewares/"
# 项目根目录 middlewares（与仓库顶层同名）
cp -a "$ROOT/middlewares/." "$PACK/middlewares/"
# 顶层 utils（与 harness 下 utils 内容一致，便于覆盖服务器上同名 utils）
cp -a "$ROOT/backend/packages/harness/deerflow/utils/." "$PACK/utils/"

TAR="/tmp/deerflow-citation-upload-$(date +%Y%m%d-%H%M%S).tar.gz"
( cd "$PACK" && tar czf "$TAR" . )
echo "打包完成: $TAR ($(du -h "$TAR" | cut -f1))"

# 优先：小文件用 ssh 管道上传（部分网络下比 scp 稳）
if ssh -o BatchMode=yes -o ConnectTimeout=30 "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR' && cat > '$REMOTE_DIR/.citation-upload.tar.gz'" < "$TAR"; then
  ssh -o BatchMode=yes "$REMOTE_HOST" "cd '$REMOTE_DIR' && tar xzf .citation-upload.tar.gz && rm -f .citation-upload.tar.gz && echo OK && du -sh utils middlewares backend/packages/harness/deerflow/utils backend/packages/harness/deerflow/agents/middlewares 2>/dev/null || true"
  echo "完成：已解压到 $REMOTE_DIR"
  exit 0
fi

echo "管道上传失败，尝试 scp …"
scp -o ServerAliveInterval=10 "$TAR" "$REMOTE_HOST:/root/deerflow-citation-upload.tar.gz"
ssh -o BatchMode=yes "$REMOTE_HOST" "cd '$REMOTE_DIR' && tar xzf /root/deerflow-citation-upload.tar.gz && rm -f /root/deerflow-citation-upload.tar.gz && echo OK"
