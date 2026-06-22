#!/usr/bin/env bash
# ===================================================================
# spark-logs.sh — Spark の app.py 処理ログを Mac のターミナルにライブ表示
# ===================================================================
# ブラウザで動画/テキストを実行すると、文字起こし・話者分離・照合・マッチングの
# 進行ログがここに流れる。Ctrl+C で監視終了（Spark の app.py は止まらない）。
#
# spark-up.sh の ControlMaster 接続が生きていればパスワード再入力なしで繋がる。
# 設定は spark-up.sh と同じ環境変数で上書き可。
# 引数で行数を指定可（既定: 直近40行から追従）。例: ./spark-logs.sh 200
# ===================================================================
set -uo pipefail

SPARK_USER=${SPARK_USER:-watalab}
SPARK_HOST=${SPARK_HOST:-147.157.219.31}
TAIL_LINES=${1:-40}

SPARK="${SPARK_USER}@${SPARK_HOST}"
CM_PATH="$HOME/.ssh/cm-%r@%h:%p"

printf '\033[1;36m[spark-logs]\033[0m %s\n' "Spark(${SPARK}) /tmp/app.log を追従中… (Ctrl+C で終了)"
exec ssh -o "ControlPath=${CM_PATH}" -o ControlMaster=auto -o ControlPersist=600 \
  -t "$SPARK" "tail -n ${TAIL_LINES} -f /tmp/app.log"
