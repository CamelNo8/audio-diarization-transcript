#!/usr/bin/env bash
# ===================================================================
# spark-down.sh — spark-up.sh で張ったトンネル / 共有SSH接続を閉じる
# ===================================================================
# 既定: トンネル（と ControlMaster 共有接続）を閉じる。Spark の app.py は残す
#       （次回 ./spark-up.sh が速い・DB編集中のセッションを切らない）。
# --stop-server: Spark 上の app.py(uvicorn) も停止する。
#
# 設定は spark-up.sh と同じ環境変数で上書き可。
# ===================================================================
set -uo pipefail

SPARK_USER=${SPARK_USER:-watalab}
SPARK_HOST=${SPARK_HOST:-147.157.219.31}
LOCAL_PORT=${LOCAL_PORT:-8001}

SPARK="${SPARK_USER}@${SPARK_HOST}"
CM_PATH="$HOME/.ssh/cm-%r@%h:%p"
SSH_OPTS=(-o "ControlPath=${CM_PATH}")

STOP_SERVER=0
[ "${1:-}" = "--stop-server" ] && STOP_SERVER=1

say() { printf '\033[1;36m[spark-down]\033[0m %s\n' "$*"; }

# 1. Spark の app.py を止める（--stop-server 指定時のみ）
if [ "$STOP_SERVER" -eq 1 ]; then
  say "Spark の app.py を停止します…"
  ssh "${SSH_OPTS[@]}" "$SPARK" "pkill -f '[.]venv/bin/python app.py'" 2>/dev/null \
    && say "停止しました。" || say "（稼働中の app.py は見つかりませんでした）"
fi

# 2. 共有 SSH 接続（ControlMaster）を閉じる → 多重化されたトンネルも一緒に落ちる
if ssh "${SSH_OPTS[@]}" -O exit "$SPARK" 2>/dev/null; then
  say "共有SSH接続とトンネルを閉じました。"
else
  # フォールバック: -fN トンネルプロセスを直接 kill
  if pkill -f "ssh.*-L ${LOCAL_PORT}:localhost" 2>/dev/null; then
    say "トンネルプロセスを停止しました (localhost:${LOCAL_PORT})。"
  else
    say "閉じるべきトンネル/共有接続は見つかりませんでした。"
  fi
fi

say "完了。"
