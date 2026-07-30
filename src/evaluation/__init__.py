"""実験（予稿 4.3 / 4.5）用のクリップ選定と評価。

字幕制作支援システム本体の処理系ではなく、**実験を回すための道具**を置く。

- :mod:`src.evaluation.srt_stats` … 正解字幕SRTの解析と説明変数の算出（共通土台）
- :mod:`src.evaluation.clip_selector` … 正解字幕から5分クリップ2本を選ぶ
- :mod:`src.evaluation.evaluator` … アプリ生成字幕を正解字幕と突き合わせて評価する
"""
