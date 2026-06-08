import csv
import argparse
from typing import List, Dict, Optional, Any

def load_subtitle_data(input_csv_path: str) -> List[Dict[str, str]]:
    """
    編集済みのCSV（対応表）ファイルを読み込み、字幕生成に必要なデータを抽出する。

    Args:
        input_csv_path (str): 入力CSVファイルのパス。

    Returns:
        List[Dict[str, str]]: 抽出されたデータのリスト。
                                各辞書には "start_time", "end_time", "speaker", "subtitle_text" が含まれる。
    """
    subtitle_data = []
    required_columns = ["start_time", "end_time", "speaker", "subtitle_text"]

    try:
        with open(input_csv_path, mode='r', encoding='utf-8-sig') as csvfile:
            # csv.DictReaderを使用してCSVを辞書として読み込む
            reader = csv.DictReader(csvfile)

            for i, row in enumerate(reader):
                # 必要な列が存在するかチェック
                if not all(col in row for col in required_columns):
                    print(f"警告: {i+1}行目に必要な列が不足しています。スキップします。")
                    continue

                # 必要な列のデータが空でないかチェック
                if not all(row[col] for col in required_columns):
                    print(f"警告: {i+1}行目に空のデータがあります。スキップします。 (speakerは空でも可)")
                    # speaker は空欄を許可する場合（仕様には明記されていないが、空の場合もありうるため）
                    if not all(row[col] for col in ["start_time", "end_time", "subtitle_text"]):
                         print(f"警告: {i+1}行目に必須データ（時間またはテキスト）がありません。スキップします。")
                         continue

                # 必要なデータのみを抽出してリストに追加
                subtitle_data.append({
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "speaker": row["speaker"],
                    "subtitle_text": row["subtitle_text"]
                })

    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_csv_path}' が見つかりません。")
        return []
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中にエラーが発生しました: {e}")
        return []

    print(f"'{input_csv_path}' から {len(subtitle_data)} 件の字幕データを読み込みました。")
    return subtitle_data

def format_subtitle_text(current_speaker: str, subtitle_text: str, previous_speaker: Optional[str]) -> str:
    """
    話者名の表示ルールに基づき、SRTに表示するテキストをフォーマットする。

    - 直前の話者と異なる場合: (話者名)字幕テキスト
    - 直前の話者と同じ場合: 字幕テキスト

    Args:
        current_speaker (str): 現在の字幕の話者。
        subtitle_text (str): 現在の字幕テキスト。
        previous_speaker (Optional[str]): 直前の字幕の話者。

    Returns:
        str: SRTに表示するフォーマット済みテキスト。
    """
    # 話者名が空欄の場合は、テキストのみを返す
    if not current_speaker:
        return subtitle_text
    # 直前の話者と比較（どちらもNoneや空文字でないことを確認）
    if previous_speaker and current_speaker == previous_speaker:
        # 話者が同じ場合はテキストのみ
        return subtitle_text
    else:
        # 話者が異なる場合、または最初の字幕の場合は話者名を追加
        return f"({current_speaker}){subtitle_text}"

def generate_srt_content(subtitle_data: List[Dict[str, str]]) -> str:
    """
    抽出された字幕データリストから、SRTファイル全体のコンテンツ（文字列）を生成する。

    Args:
        subtitle_data (List[Dict[str, str]]): load_subtitle_dataから返されたデータのリスト。

    Returns:
        str: SRTファイル形式の全コンテンツ。
    """
    srt_blocks = []
    previous_speaker: Optional[str] = None

    for i, data in enumerate(subtitle_data, 1):
        subtitle_index = i
        start_time = data["start_time"]
        end_time = data["end_time"]
        speaker = data["speaker"].strip() # 前後の空白を除去
        subtitle_text = data["subtitle_text"]
        # タイムスタンプ行をフォーマット
        timestamp_line = f"{start_time} --> {end_time}"
        # 表示テキストをフォーマット
        text_to_display = format_subtitle_text(speaker, subtitle_text, previous_speaker)
        # 現在の話者を「直前の話者」として保存
        # 話者が空欄だった場合は、比較対象にならないよう None を維持する
        if speaker:
            previous_speaker = speaker
        else:
            # 話者名が空欄の場合、次の字幕は必ず話者名を表示する必要があるため
            # previous_speaker をリセットする
            previous_speaker = None
        # SRTブロックを作成
        block = f"{subtitle_index}\n{timestamp_line}\n{text_to_display}\n"
        srt_blocks.append(block)
    # 全てのブロックを改行で結合して返す
    return "\n".join(srt_blocks)

def write_srt_file(output_srt_path: str, srt_content: str) -> None:
    """
    生成されたSRTコンテンツを指定されたファイルパスに書き込む。

    Args:
        output_srt_path (str): 出力SRTファイルのパス。
        srt_content (str): 書き込むSRTコンテンツ文字列。
    """
    try:
        with open(output_srt_path, mode='w', encoding='utf-8') as f:
            f.write(srt_content)
        print(f"SRTファイルを '{output_srt_path}' に正常に書き込みました。")
    except IOError as e:
        print(f"エラー: ファイルの書き込み中にエラーが発生しました: {e}")
    except Exception as e:
        print(f"エラー: 予期せぬエラーが発生しました: {e}")

def main():
    """
    メイン処理。コマンドライン引数を解析し、CSVからSRTへの変換を実行する。
    """
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="編集済みの対応表CSVファイルからSRT字幕ファイルを生成します。")
    parser.add_argument(
        "input_csv",
        type=str,
        help="入力する編集済みCSVファイルのパス (例: 対応表_edited.csv)"
    )
    parser.add_argument(
        "output_srt",
        type=str,
        help="出力するSRTファイルのパス (例: subtitles_exported.srt)"
    )

    args = parser.parse_args()
    # 1. CSVデータの読み込み
    subtitle_data = load_subtitle_data(args.input_csv)

    if not subtitle_data:
        print("データが読み込めなかったため、処理を終了します。")
        return
    # 2. SRTコンテンツの生成
    srt_content = generate_srt_content(subtitle_data)
    # 3. SRTファイルへの書き込み
    write_srt_file(args.output_srt, srt_content)

if __name__ == "__main__":
    main()
