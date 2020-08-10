"""
bestモデルで予測
- 翌日の終値が最終日の5%以上低ければクラス「1」
- 翌日の終値が最終日の5%以上高ければクラス「2」
- それ以外はクラス「0」
Usage:
    # スズキについて
    $ python tf_predict_best_model.py -c 7269 --is_2day_label

    # 入力画像の最終日を指定
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10 --is_2day_label

    # 数日さかのぼって実行。下記は10日さかのぼる（2020/06/01から2020/06/10を最終日として10日間毎日予測）
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10 -t_d 10 --is_2day_label

    # JPX日経インデックス400 + 日経225 + 日経500種について
    $ python tf_predict_best_model.py --is_2day_label

    # バイナリラベルのモデルで予測
    $ python tf_predict_best_model.py -c 7269 --is_binary_label -m D:\work\candlestick_model\output\model\ts_dataset_all_test_binary\best_val_loss.h5 -o D:\work\candlestick_model\output\model\ts_dataset_all_test_binary\predict
    $ python tf_predict_best_model.py -c 7269 --is_2day_label --is_binary_label -m D:\work\candlestick_model\output\model\ts_dataset_all_2day_label_binary\best_val_loss.h5 -o D:\work\candlestick_model\output\model\ts_dataset_all_2day_label_binary\predict

    # 3クラスのモデルで予測
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10 -m D:\work\candlestick_model\output\model\ts_dataset_all_test\best_val_loss.h5 -o D:\work\candlestick_model\output\model\ts_dataset_all_test\predict
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10 -t_d 10 -m D:\work\candlestick_model\output\model\ts_dataset_all_test\best_val_loss.h5 -o D:\work\candlestick_model\output\model\ts_dataset_all_test\predict

    # 4クラスのモデルで予測
    $ python tf_predict_best_model.py -c 7269 -d 2020-06-10 -t_d 10 -m D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\best_val_loss.h5 -o D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\predict --is_positive_negative_line_label

"""
import os
import sys
import glob
import sqlite3
import pandas as pd
import datetime
import traceback
import argparse
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# tensorflowのINFOレベルのログを出さないようにする
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

keras_py_path = r"C:\Users\81908\jupyter_notebook\tfgpu_py36_work\02_keras_py"
sys.path.append(keras_py_path)
from predicter import tf_grad_cam as grad_cam
from predicter import tf_base_predict as base_predict

sys.path.append(
    r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\code"
)
import make_candlestick


class Predict:
    def __init__(
        self,
        code,
        start_date,
        end_date,
        label_col,
        is_2day_label,
        is_positive_negative_line_label,
    ):
        self.code = int(code)
        self.start_date = start_date
        self.end_date = end_date
        self.label_col = label_col
        self.is_2day_label = is_2day_label
        self.is_positive_negative_line_label = is_positive_negative_line_label

    def get_pred_df(self, day_period=20):
        """
        予測用に1銘柄について画像の元となるデータフレーム取得
        """
        df = make_candlestick.get_code_prices(
            self.code, str(self.start_date), str(self.end_date)
        )
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.set_index("date")
        df.columns = ["Code", "Open", "High", "Low", "Close", "Volume"]
        df = df.tail(day_period)

        try:
            # 翌営業のレコードあればラベル取得 祝日はさむ場合に備えて多めにとる
            _end_date = df.iloc[-1].name.date()
            # print("_end_date:", _end_date)
            df_label = make_candlestick.get_code_prices(
                code, str(_end_date), str(self.end_date + datetime.timedelta(days=15))
            )
            # print(df_label)
            # print()
            if self.is_positive_negative_line_label:
                # is_positive_negative_line_label の場合は始値終値の値とるので
                label_price = (
                    str(df_label.iloc[1]["open"])
                    + "-"
                    + str(df_label.iloc[1]["close"])
                    + f" ({df_label.iloc[1]['close'] - df_label.iloc[1]['open']})"
                )
                # print(label_price)
                # print()
            elif self.is_2day_label:
                # is_2day_labelの場合は明日、明後日の値とるので
                label_price = (
                    str(df_label.iloc[1][self.label_col])
                    + ", "
                    + str(df_label.iloc[2][self.label_col])
                )
            else:
                # print(df_label.iloc[1])
                label_price = df_label.iloc[1][self.label_col]
        except Exception:
            # traceback.print_exc()
            label_price = 0.0  # None
        # print(df, df.shape, label_price)
        return df, label_price

    def pred_candlestick(
        self, model, output_png=None, classes=["0", "1"], img_size=[80, 80, 3],
    ):
        """ローソク足の画像作成して予測+gradcam"""
        # ラベルと株価取得
        df, label_price = self.get_pred_df()

        # 画像出力
        if self.is_2day_label or len(classes) == 4:
            make_candlestick.make_mav_candlestick_mplfinance(df, output_png)
        else:
            make_candlestick.make_candlestick_mplfinance(df, output_png)

        # 予測
        y, pb = base_predict.pred_from_1img(
            model, output_png, img_size[0], img_size[1], classes=classes, show_img=True
        )

        # gradcam
        _ = grad_cam.image2gradcam(model, output_png)

        return y, pb, label_price, df


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=r"D:\work\candlestick_model\output\model\ts_dataset_all_2day_label\Xception\predict",
    )
    ap.add_argument(
        "-m",
        "--model_path",
        type=str,
        default=r"D:\work\candlestick_model\output\model\ts_dataset_all_2day_label\Xception\best_val_loss.h5",
    )
    ap.add_argument("-c", "--codes", type=int, nargs="*", default=None)
    ap.add_argument("-d", "--date_exe", type=str, default=None)
    ap.add_argument("-t_d", "--term_days", type=int, default=1)
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
        help="翌日、明後日の値ラベルにするか",
    )
    ap.add_argument(
        "-is_pn",
        "--is_positive_negative_line_label",
        action="store_const",
        const=True,
        default=False,
        help="翌日が陽線、陰線か",
    )
    return vars(ap.parse_args())


if __name__ == "__main__":
    matplotlib.use("Agg")

    args = get_args()

    output_img_dir = os.path.join(args["output_dir"], "image")
    os.makedirs(output_img_dir, exist_ok=True)

    if args["is_binary_label"]:
        label_names = ["0", "1"]
        label_col = "low"
        date_exe_col = "High"
    elif args["is_2day_label"]:
        label_names = ["0", "1", "2"]
        label_col = "open"
        date_exe_col = "Close"
    elif args["is_positive_negative_line_label"]:
        label_names = ["0", "1", "2", "3"]
        label_col = None
        date_exe_col = "Close"
    else:
        label_names = ["0", "1", "2"]
        label_col = "close"
        date_exe_col = "Close"

    # 前実行したgradcam画像削除
    _ = [os.remove(p) for p in glob.glob(os.path.join(output_img_dir, "*gradcam.jpg"))]

    # モデルロード
    model = keras.models.load_model(args["model_path"], compile=False)

    date_exes = []
    cs = []
    ys = []
    pbs = []
    exe_prices = []
    label_prices = []
    # args['term_days']で指定した日数前から実行繰り返す
    for t_d in range(args["term_days"]):

        # 実行日の指定なければ、入力画像の最終日を今日にする
        if args["date_exe"] is None:
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            d_end_date = datetime.datetime.strptime(today, "%Y-%m-%d").date()
        else:
            d_end_date = datetime.datetime.strptime(args["date_exe"], "%Y-%m-%d").date()

        # t_d日戻る
        date_exe = d_end_date - datetime.timedelta(days=t_d)

        d_start_date = date_exe - datetime.timedelta(days=20 * 3)
        print(d_start_date, date_exe)

        if args["codes"] is None:
            # JPX日経インデックス400 + 日経225 + 日経500種
            # https://indexes.nikkei.co.jp/nkave/index/component?idx=nk500av
            with open(
                r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\jpx400_nikkei225_500.txt"
            ) as fp:
                codes = [c.strip() for c in fp.readlines()]  # stripで両端の文字を削除
        else:
            codes = args["codes"]

        for code in codes:
            try:
                pred_cls = Predict(
                    code,
                    d_start_date,
                    date_exe,
                    label_col,
                    args["is_2day_label"],
                    args["is_positive_negative_line_label"],
                )

                # 画像作成して予測
                output_png = os.path.join(
                    output_img_dir, f"{str(code)}_candlestick.png"
                )
                y, pb, label_price, df = pred_cls.pred_candlestick(
                    model, output_png=output_png, classes=label_names,
                )

                # 実行日とその値
                _date_exe = str(df.iloc[-1].name.date())
                _date_exe_price = df.iloc[-1][date_exe_col]

                print(
                    "pred_class, probability, date_exe, date_exe_high, label_price:\n",
                    y,
                    pb,
                    _date_exe,
                    _date_exe_price,
                    label_price,
                )

                date_exes.append(str(_date_exe))
                cs.append(code)
                ys.append(y)
                pbs.append(pb)
                exe_prices.append(_date_exe_price)
                label_prices.append(label_price)

            except Exception:
                traceback.print_exc()
                pass

    if args["is_2day_label"]:
        # print("label_prices:", label_prices)
        label_prices = ["0.0, 0.0" if p == 0.0 else p for p in label_prices]
        # print("label_prices:", label_prices)
        pred_df = pd.DataFrame(
            {
                "date_exe": date_exes,
                "code": cs,
                "pred_pb": pbs,
                "2day_pred_y": ys,
                "date_exe_price": exe_prices,
                "tomorrow_true_price": map(
                    lambda x: float(x.split(", ")[0]), label_prices
                ),
                "day_after_tomorrow_true_price": map(
                    lambda x: float(x.split(", ")[1]), label_prices
                ),
            }
        )
    else:
        pred_df = pd.DataFrame(
            {
                "date_exe": date_exes,
                "code": cs,
                "pred_pb": pbs,
                "next_day_pred_y": ys,
                "date_exe_price": exe_prices,
                "next_day_true_price": label_prices,
            }
        )
    pred_df = pred_df.sort_values(by=["date_exe", "code"])
    pred_df = pred_df.drop_duplicates().reset_index(drop=True)
    pred_df.to_excel(os.path.join(args["output_dir"], "pred.xlsx"))
    pred_df.to_csv(os.path.join(args["output_dir"], "pred.csv"), index=False)
