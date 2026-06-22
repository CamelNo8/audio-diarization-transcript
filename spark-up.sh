#!/usr/bin/env bash
# ===================================================================
# spark-up.sh — Mac から1コマンドで「Spark の app.py」を起動・接続する
# ===================================================================
# やること:
#   1. Spark 上で app.py(uvicorn 127.0.0.1:8000) をバックグラウンド起動
#      （大学プロキシ ntut ＋ HF_HUB_OFFLINE=0 付き / 既に起動中なら再起動しない）
#   2. Mac→Spark の SSH トンネル ($LOCAL_PORT:localhost:$REMOTE_PORT) を確立
#   3. サーバー応答を待ってから Mac のブラウザで http://127.0.0.1:$LOCAL_PORT/ を開く
#
# SSH パスワード入力は ControlMaster(接続多重化) で原則1回だけ。
# 終了は ./spark-down.sh（トンネル切断。--stop-server で Spark の app.py も停止）。
#
# 設定は環境変数で上書き可:
#   SPARK_USER / SPARK_HOST / REMOTE_REPO / REMOTE_PORT / LOCAL_PORT
# 例: LOCAL_PORT=8002 ./spark-up.sh
# ===================================================================
set -uo pipefail

SPARK_USER=${SPARK_USER:-watalab}
SPARK_HOST=${SPARK_HOST:-147.157.219.31}
REMOTE_REPO=${REMOTE_REPO:-'~/yamada/dev/audio-diarization-transcript'}
REMOTE_PORT=${REMOTE_PORT:-8000}
LOCAL_PORT=${LOCAL_PORT:-8001}

SPARK="${SPARK_USER}@${SPARK_HOST}"
CM_PATH="$HOME/.ssh/cm-%r@%h:%p"
# ControlMaster: 1本の SSH 接続を多重化し、以降の ssh はパスワード再入力なしにする。
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CM_PATH}" -o ControlPersist=600)

mkdir -p "$HOME/.ssh"

say() { printf '\033[1;36m[spark-up]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[spark-up]\033[0m %s\n' "$*" >&2; }

# ------------------------------------------------------------------
# 1. Spark 上で app.py を起動（既に :$REMOTE_PORT が応答するなら何もしない）
# ------------------------------------------------------------------
# bash -ic（インタラクティブ）で ~/.bashrc を読み込み、ユーザー定義の ntut 関数を有効化する。
# ~/.bashrc の非対話 early-return ガードで ntut が見つからない場合は、下のフォールバック行
# （プロキシURLを直接 export）に切り替えてください（memory: NTUT gw）。
REMOTE_CMD="cd ${REMOTE_REPO} && ntut && export HF_HUB_OFFLINE=0 && \
(curl -fsS http://127.0.0.1:${REMOTE_PORT}/ >/dev/null 2>&1 \
  || nohup .venv/bin/python app.py >/tmp/app.log 2>&1 &); sleep 1"
# --- フォールバック（ntut が使えない場合はこの行を上の ntut 行と差し替え）---
# REMOTE_CMD="cd ${REMOTE_REPO} && export https_proxy=http://gw.a.tsukuba-tech.ac.jp:8080 http_proxy=http://gw.a.tsukuba-tech.ac.jp:8080 no_proxy=localhost,127.0.0.1 HF_HUB_OFFLINE=0 && (curl -fsS http://127.0.0.1:${REMOTE_PORT}/ >/dev/null 2>&1 || nohup .venv/bin/python app.py >/tmp/app.log 2>&1 &); sleep 1"

say "Spark(${SPARK}) で app.py を起動中…（初回はパスワードを入力）"
if ! ssh "${SSH_OPTS[@]}" -tt "$SPARK" "bash -ic '${REMOTE_CMD}'"; then
  err "Spark への SSH / app.py 起動に失敗しました。"
  exit 1
fi

# ------------------------------------------------------------------
# 2. SSH トンネルを張る（既に LOCAL_PORT が LISTEN 中ならスキップ）
# ------------------------------------------------------------------
if lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  say "トンネルは既に存在します (localhost:${LOCAL_PORT})。再利用します。"
else
  say "トンネルを確立: localhost:${LOCAL_PORT} → ${SPARK_HOST}:${REMOTE_PORT}"
  if ! ssh "${SSH_OPTS[@]}" -fN -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "$SPARK"; then
    err "トンネル確立に失敗しました。"
    exit 1
  fi
fi

# ------------------------------------------------------------------
# 3. サーバー応答を待つ（モデルロードで初回は時間がかかる）→ ブラウザを開く
# ------------------------------------------------------------------
URL="http://127.0.0.1:${LOCAL_PORT}/"   # 必ず http://（https 自動昇格だと接続拒否になる）
say "サーバー応答待ち… (${URL})"
for i in $(seq 1 30); do
  if curl -fsS "$URL" >/dev/null 2>&1; then
    say "起動完了。ブラウザを開きます: ${URL}"
    open "$URL"
    say "終了するには: ./spark-down.sh （Spark の app.py も止めるなら --stop-server）"
    exit 0
  fi
  sleep 1
done

err "サーバーが ${URL} で応答しませんでした。Spark 側のログを確認します:"
ssh "${SSH_OPTS[@]}" "$SPARK" "tail -n 40 /tmp/app.log" 2>/dev/null || true
err "（トンネルは張られています。手動で ${URL} を開くか、ログを確認してください）"
exit 1
