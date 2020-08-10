"""
全銘柄コードについてローソク足の画像作成
- https://www.arxiv-vanity.com/papers/1903.12258/
Usage:
    $ conda activate tfgpu

    # 3クラスで出力
    $ python make_candlestick.py

    # バイナリラベルで出力
    $ python make_candlestick.py --is_binary_label -o D:\work\candlestick_model\output\orig_image_all_binary

    # 翌日、明後日の値でラベルづけ + 移動平均線つき
    $ python make_candlestick.py --is_mav_png --is_2day_label -o D:\work\candlestick_model\output\orig_image_all_2day_label

    # 翌日が陽線、陰線か + 移動平均線つき
    $ python make_candlestick.py --is_mav_png --is_positive_negative_line_label -o D:\work\candlestick_model\output\orig_image_all_positive_negative_line

    # テスト用
    $ python make_candlestick.py -o D:\work\candlestick_model\output\tmp
"""
import os
import glob
import gc
import sqlite3
import datetime
import argparse
import traceback
import pathlib
from tqdm import tqdm

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def table_to_df(
    table_name=None, sql=None, db_file_name=r"D:\DB_Browser_for_SQLite\stock.db"
):
    """sqlite3で指定テーブルのデータをDataFrameで返す"""
    conn = sqlite3.connect(db_file_name)
    if table_name is not None:
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    elif sql is not None:
        return pd.read_sql(sql, conn)
    else:
        return None


def get_code_prices(code, start_date, end_date):
    """DBから指定銘柄の株価取得"""
    sql = f"""
    SELECT
        *
    FROM
        prices AS t
    WHERE
        t.code = {code}
    AND
        t.date >= '{start_date}'
    AND
        t.date <= '{end_date}'
    """
    return table_to_df(sql=sql)


def make_candlestick_mplfinance(df, save_png: str) -> None:
    """
    mplfinanceを使ってローソク足画像作成する
    ※論文では出来高入れない方が精度良さそうだったから出来高は入れない
    """
    import sys

    current_dir = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(current_dir) + "/../GitHub/mplfinance/src")
    import mplfinance as mpf

    kwargs = dict(
        type="candle",
        figratio=(1.5, 1.5),
        figscale=1,
        title="",
        ylabel="",
        ylabel_lower="",
    )
    mc = mpf.make_marketcolors(up="#00ff00", down="#ff00ff", inherit=True)
    s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridstyle="")
    mpf.plot(
        df, **kwargs, style=s, is_not_set_visible=True, savefig=save_png,
    )


def make_mav_candlestick_mplfinance(df, save_png: str) -> None:
    """
    mplfinanceを使って移動平均線+ローソク足画像作成する
    ※論文では出来高入れない方が精度良さそうだったから出来高は入れない
    """
    import sys

    current_dir = pathlib.Path(__file__).resolve().parent
    sys.path.append(str(current_dir) + "/../GitHub/mplfinance/src")
    import mplfinance as mpf

    kwargs = dict(
        type="candle",
        figratio=(1.5, 1.5),
        figscale=1,
        title="",
        ylabel="",
        ylabel_lower="",
        mav=(5, 10),
    )
    mc = mpf.make_marketcolors(up="#FF0000", down="#0000FF", inherit=True)
    s = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc, gridstyle="")
    mpf.plot(
        df, **kwargs, style=s, is_not_set_visible=True, savefig=save_png,
    )


def get_label_df(code: int, start_date, day_period=20):
    """
    1銘柄についてラベルと画像の元となるデータフレーム取得
    day_periodの翌日の終値でラベルづけする
    """
    try:
        # day_period以上レコード必要なので少し先までデータ取得
        end_date = start_date + datetime.timedelta(days=day_period * 3)
        # 期間なければ例外飛ぶからtryで囲んでる
        df = get_code_prices(code, str(start_date), str(end_date))
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.set_index("date")
        df.columns = ["Code", "Open", "High", "Low", "Close", "Volume"]

        label = None
        # day_period+1レコードなければチャート出さない
        if df.shape[0] >= day_period + 1:
            df = df.iloc[0 : day_period + 1]
            label_price = df.iloc[-1]["Close"]
            # print("label_price:", label_price)
            last_price = df.iloc[-2]["Close"]
            df = df.head(day_period)

            label = None
            if last_price * 0.95 >= label_price:
                # 翌日の終値が最終日の5%以上低ければ「1」
                label = 1
            elif last_price * 1.05 <= label_price:
                # 翌日の終値が最終日の5%以上高ければ「2」
                label = 2
            else:
                # 0.95-1.05の間なら「0」
                label = 0
        return df, label

    except Exception as e:
        traceback.print_exc()
        return None, None


def get_2day_label_df(code: int, start_date, day_period=20, is_debug=False):
    """
    1銘柄についてラベルと画像の元となるデータフレーム取得
    day_periodの翌日、明後日の値でラベルづけする
    """
    try:
        # day_period以上レコード必要なので少し先までデータ取得
        end_date = start_date + datetime.timedelta(days=day_period * 10)
        # 期間なければ例外飛ぶからtryで囲んでる
        df = get_code_prices(code, str(start_date), str(end_date))
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.set_index("date")
        df.columns = ["Code", "Open", "High", "Low", "Close", "Volume"]

        label = None
        # day_period+2レコードなければチャート出さない
        if df.shape[0] >= day_period + 2:
            df = df.iloc[0 : day_period + 2]
            last_price = df.iloc[-3]["Close"]
            label_1_open = df.iloc[-2]["Open"]
            label_2_open = df.iloc[-1]["Open"]
            if is_debug:
                print(last_price, label_1_open, label_2_open)
                print(df)

            # 画像に使うデータはday_periodの期間だけ
            df = df.head(day_period)

            label = None
            if label_1_open < label_2_open:
                # 2日目が1日目よりがってたら「1」
                label = 1
                if label_1_open * 1.05 <= label_2_open:
                    # 2日目が1日目より5%以上がっても「2」
                    label = 2
            elif (last_price * 0.95 >= label_1_open) and (last_price < label_2_open):
                # 5%以上V字回復なら「2」
                label = 2
            else:
                # それ以外なら「0」
                label = 0
        return df, label

    except Exception as e:
        traceback.print_exc()
        return None, None


def get_positive_negative_line_label_df(
    code: int, start_date, day_period=20, is_debug=False
):
    """
    1銘柄についてラベルと画像の元となるデータフレーム取得
    day_periodの翌日が陽線、陰線かどうかでラベルつける
    """
    try:
        # day_period以上レコード必要なので少し先までデータ取得
        end_date = start_date + datetime.timedelta(days=day_period * 10)
        # 期間なければ例外飛ぶからtryで囲んでる
        df = get_code_prices(code, str(start_date), str(end_date))
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.set_index("date")
        df.columns = ["Code", "Open", "High", "Low", "Close", "Volume"]

        label = None
        # day_period+1レコードなければチャート出さない
        if df.shape[0] >= day_period + 1:
            df = df.iloc[0 : day_period + 1]

            # ラベルとなる行
            label_row = df.iloc[-1]
            # ローソク足
            candle = label_row["Close"] - label_row["Open"]
            # 線
            line = label_row["High"] - label_row["Low"]

            if candle <= 0.0:
                # 陰線なら「0」
                label = 0
                if abs(candle) / line >= 0.7:
                    # 大陰線（ローソク足が7割占めるときとした）なら「1」
                    label = 1
            else:
                # 陽線なら「2」
                label = 2
                if abs(candle) / line >= 0.7:
                    # 大陽線（ローソク足が7割占めるときとした）なら「3」
                    label = 3

            if is_debug:
                print(candle, line, label)
                print(df)

            # 画像に使うデータはday_periodの期間だけ
            df = df.head(day_period)

        return df, label

    except Exception as e:
        traceback.print_exc()
        return None, None


def get_binary_label_df(code: int, start_date, day_period=20):
    """
    1銘柄についてラベルと画像の元となるデータフレーム取得
    day_periodの翌日の終値でバイナリでラベルづけする
    """
    try:
        # day_period以上レコード必要なので少し先までデータ取得
        end_date = start_date + datetime.timedelta(days=day_period * 3)
        # 期間なければ例外飛ぶからtryで囲んでる
        df = get_code_prices(code, str(start_date), str(end_date))
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.set_index("date")
        df.columns = ["Code", "Open", "High", "Low", "Close", "Volume"]

        label = None
        # day_period+1レコードなければチャート出さない
        if df.shape[0] >= day_period + 1:
            df = df.iloc[0 : day_period + 1]
            label_price = df.iloc[-1]["Low"]
            # print("label_price:", label_price)
            last_price = df.iloc[-2]["High"]
            df = df.head(day_period)

            label = None
            if last_price > label_price:
                # 最終日よりも低ければ「0」
                label = 0
            elif last_price < label_price:
                # 最終日よりも高ければ「1」
                label = 1

        return df, label

    except Exception as e:
        traceback.print_exc()
        return None, None


def daterange(start_date, end_date):
    """
    2つのdatetimeオブジェクトがあって、その期間の1日ごとに処理を行うgenerater
    http://y0m0r.hateblo.jp/entry/20120122/1327217763
    Usage:
        import datetime
        start = datetime.datetime.strptime('201201', '%Y%m')
        end = datetime.datetime.strptime('201202', '%Y%m')
        for i in daterange(start, end):
            # 処理
            print(i)
    """
    for n in range((end_date - start_date).days):
        yield start_date + datetime.timedelta(n)


def gen_file_name(path):
    """ファイル名返すgenerator"""
    import glob
    import pathlib

    for p in glob.glob(path):
        yield pathlib.Path(p).stem


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input_txt",
        type=str,
        default=r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\nikkei1000.txt",
    )
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=r"D:\work\candlestick_model\output\orig_image_all",
    )
    ap.add_argument("-start_d", "--start_date", type=str, default="2000-01-01")
    ap.add_argument("-stop_d", "--stop_date", type=str, default="2020-07-18")
    ap.add_argument("-d_p", "--day_period", type=int, default=20)
    ap.add_argument(
        "-is_b",
        "--is_binary_label",
        action="store_const",
        const=True,
        default=False,
        help="バイナリラベルにするか",
    )
    ap.add_argument(
        "-is_2d",
        "--is_2day_label",
        action="store_const",
        const=True,
        default=False,
        help="翌日、明後日の値でラベルづけする",
    )
    ap.add_argument(
        "-is_pn",
        "--is_positive_negative_line_label",
        action="store_const",
        const=True,
        default=False,
        help="翌日が陽線、陰線か",
    )
    ap.add_argument(
        "-is_mav",
        "--is_mav_png",
        action="store_const",
        const=True,
        default=False,
        help="画像に5,10日の移動平均線入れるか",
    )
    return vars(ap.parse_args())


def main():
    args = get_args()

    output_dir = args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    ## 全銘柄コード
    # codes = [pathlib.Path(p).stem for p in glob.glob(r"D:\DB_Browser_for_SQLite\csvs\kabuoji3\*csv")]
    ## テスト用
    # codes = ["1301", "7974", "9613"]
    # for code in tqdm(codes):
    # generatorで全銘柄コード
    # for code in gen_file_name(r"D:\DB_Browser_for_SQLite\csvs\kabuoji3\*csv"):
    # 日経1000(https://indexes.nikkei.co.jp/nkave/index/component?idx=nkj1000)の銘柄について
    with open(args["input_txt"]) as fp:
        for line in tqdm(fp):
            code = line.strip()  # stripで両端の文字を削除

            start_date = args["start_date"]
            d_start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
            stop_date = args["stop_date"]
            d_stop_date = datetime.datetime.strptime(stop_date, "%Y-%m-%d").date()

            # 全営業日取得
            df_start_stop = get_code_prices(code, str(d_start_date), str(d_stop_date))
            df_start_stop["date"] = pd.to_datetime(
                df_start_stop["date"], format="%Y-%m-%d"
            )
            # print(df_start_stop["date"])

            label_1_ago = None
            # 1日ごとに処理
            for d in df_start_stop["date"]:
                d = d.date()
                d_end = d + datetime.timedelta(days=args["day_period"])

                # この日以降になったら終わらす
                if d_end >= d_stop_date:
                    break

                try:
                    # ラベルと株価取得
                    if args["is_binary_label"]:
                        df, label = get_binary_label_df(
                            int(code), d, day_period=args["day_period"]
                        )
                    elif args["is_2day_label"]:
                        df, label = get_2day_label_df(
                            int(code), d, day_period=args["day_period"]
                        )
                    elif args["is_positive_negative_line_label"]:
                        df, label = get_positive_negative_line_label_df(
                            int(code), d, day_period=args["day_period"]
                        )
                    else:
                        df, label = get_label_df(
                            int(code), d, day_period=args["day_period"]
                        )

                    # 取得失敗したら飛ばす
                    if df is None or label is None:
                        print(
                            "ラベルと株価取得失敗:", code, d, d_end,
                        )
                        continue

                    # day_periodに相当する実際の最終日
                    d_end_new = str(df.iloc[-1].name.date())

                    # 画像出力先準備
                    _out_dir = os.path.join(output_dir, str(label))
                    os.makedirs(_out_dir, exist_ok=True)
                    save_png = os.path.join(
                        _out_dir, f"{str(code)}_{str(d)}_{d_end_new}.png"
                    )

                    # 画像なければ作成
                    if os.path.exists(save_png) == False:
                        # 1日前と違うラベルなら出力（同じような画像ができ続けるのを避けるため）
                        if label_1_ago != label:
                            # 画像に移動平均線つけるか
                            if args["is_mav_png"]:
                                make_mav_candlestick_mplfinance(df, save_png)
                            else:
                                make_candlestick_mplfinance(df, save_png)
                            print(code, d, d_end_new, label)

                    # 1日前のラベル
                    label_1_ago = label

                    ## なんかメモリどんどん食われるので重そうな変数解放してみる
                    # 処理めっちゃ遅くなるからやめる
                    # 画像出力するとメモリ取らるみたい → mplfinance のplotting.py いじってメモリ解放するようにした
                    # https://teratail.com/questions/83319
                    # del df
                    # gc.collect()

                except Exception:
                    traceback.print_exc()
                    pass


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
