"""
ローソク足の画像をtrain/validation/test setに分ける
Usage:
    # 日経1000の全銘柄について validation  setあり
    $ python make_dataset.py -o D:\work\candlestick_model\output\ts_dataset_all_val_test

    # validation  setなし
    $ python make_dataset.py -o D:\work\candlestick_model\output\ts_dataset_all_test --is_test_only

    # バイナリラベル
    $ python make_dataset.py -i D:\work\candlestick_model\output\orig_image_all_binary -o D:\work\candlestick_model\output\ts_dataset_all_test_binary --is_test_only --is_binary_label

    # 翌日、明後日の値でラベルづけ + 移動平均線つき 不均衡補正なし
    $ python make_dataset.py -i D:\work\candlestick_model\output\orig_image_all_2day_label -o D:\work\candlestick_model\output\ts_dataset_all_2day_label --is_test_only --is_not_equalize

    # 翌日が陽線、陰線か + 移動平均線つき
    $ python make_dataset.py -i D:\work\candlestick_model\output\orig_image_all_positive_negative_line -o D:\work\candlestick_model\output\ts_dataset_all_positive_negative_line_label --is_test_only

    # 数減らしたテスト用
    $ python make_dataset.py -o D:\work\candlestick_model\output\ts_dataset_small -l_s 1000
    $ python make_dataset.py -i D:\work\candlestick_model\output\orig_image_all_2day_label -o D:\work\candlestick_model\output\ts_dataset_all_2day_label_small --is_test_only -l_s 1000 --is_test_equalize
    $ python make_dataset.py -i D:\work\candlestick_model\output\orig_image_all_positive_negative_line -o D:\work\candlestick_model\output\ts_dataset_all_positive_negative_line_label_small -ls 0 1 2 3 -l_s 3000 --is_test_only --is_test_equalize

    # 画像ファイルはコピーしない
    $ python make_dataset.py -o D:\work\candlestick_model\output\ts_dataset_small -l_s 1000 --is_not_copy_png
"""
import argparse
import os
import glob
import pathlib
import random
import shutil

from multiprocessing import Pool, cpu_count  # マルチプロセス（別のCPUコアの別のpythonプロセスで複数の処理を同時にやる）

# from multiprocessing.dummy import Pool  # マルチスレッド

import numpy as np
import pandas as pd
from tqdm import tqdm

seed = 42  # 乱数シード固定
random.seed(seed)
np.random.seed(seed)


def _equalize_label_size(df, label_names, l_size=None):
    """
    アンダーサンプリングでラベルごとのレコード数均一にする
    l_sizeに指定あればその数だけにする
    """
    df_l = None
    for l_n in label_names:
        idxs = df.query(f"label == '{l_n}'").index.to_list()
        # 数指定するか
        l_size = df["label"].value_counts().min() if l_size is None else l_size
        # ランダムサンプリングして各クラスのファイル数合わせる
        random_idxs = random.sample(idxs, l_size)
        df_l = (
            pd.concat([df_l, df.loc[random_idxs]])
            if df_l is not None
            else df.loc[random_idxs]
        )
    return df_l


def make_ts_dataset_train_test(
    orig_data_dir,
    dataset_dir,
    label_names,
    l_size=None,
    is_equalize=True,
    is_test_equalize=False,
    th_year=2017,
    is_copy=True,
) -> None:
    """時系列の分け方でteain/test setのファイルを分ける"""
    # ファイルパスやラベルの情報をデータフレームにする
    pngs = glob.glob(orig_data_dir + "/*/*png")
    labels = [pathlib.Path(p).parents[0].stem for p in pngs]
    years = [pathlib.Path(p).stem.split("_")[1].split("-")[0] for p in pngs]
    df_pngs = pd.DataFrame({"label": labels, "year": years, "png": pngs})

    # 2017年でそれぞれ分けることにする
    df_train = df_pngs.query(f"year <= '{th_year}'").reset_index(drop=True)
    df_test = df_pngs.query(f"year > '{th_year}' and year <= '2020'").reset_index(
        drop=True
    )

    # ちゃんと分かれたか確認
    print("--- train year value_counts ---")
    print(df_train["year"].value_counts())
    print("--- test year value_counts ---")
    print(df_test["year"].value_counts())

    if is_equalize:
        # アンダーサンプリングでラベルごとのレコード数均一にする
        # print(label_names, l_size)
        df_train = _equalize_label_size(df_train, label_names, l_size=l_size)

        # test setはそのまま使うか
        if is_test_equalize:
            df_test = _equalize_label_size(df_test, label_names, l_size=l_size)

        # ちゃんと分かれたか確認
        # display(df_train)
        # display(df_test)
        print("--- train year value_counts ---")
        print(df_train["year"].value_counts())
        print("--- test year value_counts ---")
        print(df_test["year"].value_counts())
    print()
    print("--- train label value_counts ---")
    print(df_train["label"].value_counts())
    print("--- test label value_counts ---")
    print(df_test["label"].value_counts())

    # データフレーム出力
    os.makedirs(os.path.join(dataset_dir), exist_ok=True)
    df_train.to_csv(os.path.join(dataset_dir, "df_train.csv"), index=False)
    df_test.to_csv(os.path.join(dataset_dir, "df_test.csv"), index=False)

    if is_copy:
        # ファイルコピー
        for l_n in label_names:
            os.makedirs(os.path.join(dataset_dir, "train", l_n), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "test", l_n), exist_ok=True)

        df_train.apply(
            lambda row: shutil.copy(
                row["png"],
                os.path.join(
                    dataset_dir,
                    "train",
                    str(row["label"]),
                    str(pathlib.Path(row["png"]).name),
                ),
            ),
            axis=1,
        )
        df_test.apply(
            lambda row: shutil.copy(
                row["png"],
                os.path.join(
                    dataset_dir,
                    "test",
                    str(row["label"]),
                    str(pathlib.Path(row["png"]).name),
                ),
            ),
            axis=1,
        )


def make_ts_dataset_train_val_test(
    orig_data_dir,
    dataset_dir,
    label_names,
    l_size=None,
    is_equalize=True,
    th_years=[2016, 2018],
    is_copy=True,
) -> None:
    """時系列の分け方でteain/validation/test setのファイルを分ける"""
    # ファイルパスやラベルの情報をデータフレームにする
    pngs = glob.glob(orig_data_dir + "/*/*png")
    labels = [pathlib.Path(p).parents[0].stem for p in pngs]
    years = [pathlib.Path(p).stem.split("_")[1].split("-")[0] for p in pngs]
    df_pngs = pd.DataFrame({"label": labels, "year": years, "png": pngs})

    # 2016/2018/2020年でそれぞれ分けることにする
    df_train = df_pngs.query(f"year <= '{th_years[0]}'").reset_index(drop=True)
    df_val = df_pngs.query(
        f"year > '{th_years[0]}' and year <= '{th_years[1]}'"
    ).reset_index(drop=True)
    df_test = df_pngs.query(f"year > '{th_years[1]}' and year <= '2020'").reset_index(
        drop=True
    )

    # ちゃんと分かれたか確認
    print("--- train year value_counts ---")
    print(df_train["year"].value_counts())
    print("--- valid year value_counts ---")
    print(df_val["year"].value_counts())
    print("--- test year value_counts ---")
    print(df_test["year"].value_counts())

    if is_equalize:
        # アンダーサンプリングでラベルごとのレコード数均一にする
        # print(label_names, l_size)
        df_train = _equalize_label_size(df_train, label_names, l_size=l_size)
        df_val = _equalize_label_size(df_val, label_names, l_size=l_size)
        ## test setはそのまま使う
        # df_test = _equalize_label_size(df_test, label_names, l_size=l_size)

        # ちゃんと分かれたか確認
        # display(df_train)
        # display(df_val)
        # display(df_test)
        print("--- train year value_counts ---")
        print(df_train["year"].value_counts())
        print("--- valid year value_counts ---")
        print(df_val["year"].value_counts())
        print("--- test year value_counts ---")
        print(df_test["year"].value_counts())
    print()
    print("--- train label value_counts ---")
    print(df_train["label"].value_counts())
    print("--- valid label value_counts ---")
    print(df_val["label"].value_counts())
    print("--- test label value_counts ---")
    print(df_test["label"].value_counts())

    # データフレーム出力
    os.makedirs(os.path.join(dataset_dir), exist_ok=True)
    df_train.to_csv(os.path.join(dataset_dir, "df_train.csv"), index=False)
    df_val.to_csv(os.path.join(dataset_dir, "df_validation.csv"), index=False)
    df_test.to_csv(os.path.join(dataset_dir, "df_test.csv"), index=False)

    if is_copy:
        # ファイルコピー
        for l_n in label_names:
            os.makedirs(os.path.join(dataset_dir, "train", l_n), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "validation", l_n), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, "test", l_n), exist_ok=True)
        df_train.apply(
            lambda row: shutil.copy(
                row["png"],
                os.path.join(
                    dataset_dir,
                    "train",
                    str(row["label"]),
                    str(pathlib.Path(row["png"]).name),
                ),
            ),
            axis=1,
        )
        df_val.apply(
            lambda row: shutil.copy(
                row["png"],
                os.path.join(
                    dataset_dir,
                    "validation",
                    str(row["label"]),
                    str(pathlib.Path(row["png"]).name),
                ),
            ),
            axis=1,
        )
        df_test.apply(
            lambda row: shutil.copy(
                row["png"],
                os.path.join(
                    dataset_dir,
                    "test",
                    str(row["label"]),
                    str(pathlib.Path(row["png"]).name),
                ),
            ),
            axis=1,
        )


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=r"D:\work\candlestick_model\output\ts_dataset_all_val_test",
    )
    ap.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default=r"D:\work\candlestick_model\output\orig_image_all",
    )
    ap.add_argument(
        "-l_s",
        "--l_size",
        type=int,
        default=None,
        help="1ラベルの画像枚数指定する.1000なら1ラベル1000枚だけにする",
    )
    ap.add_argument(
        "-is_t_o",
        "--is_test_only",
        action="store_const",
        const=True,
        default=False,
        help="validation set作らずtest setだけにするか",
    )
    ap.add_argument(
        "-is_b",
        "--is_binary_label",
        action="store_const",
        const=True,
        default=False,
        help="バイナリラベルにするか",
    )
    ap.add_argument(
        "-ls",
        "--labels",
        nargs="*",
        type=str,
        default=None,
        help="任意のラベル. ex: 0 1 2 3",
    )
    ap.add_argument(
        "-is_n_e",
        "--is_not_equalize",
        action="store_const",
        const=True,
        default=False,
        help="ラベル均等にしないか",
    )
    ap.add_argument(
        "-is_t_e",
        "--is_test_equalize",
        action="store_const",
        const=True,
        default=False,
        help="test setのラベル均等にする",
    )
    ap.add_argument(
        "-is_n_c",
        "--is_not_copy_png",
        action="store_const",
        const=True,
        default=False,
        help="pngファイルコピーしないか",
    )
    return vars(ap.parse_args())


if __name__ == "__main__":
    args = get_args()

    # ラベル指定
    label_names = ["0", "1", "2"]
    if args["is_binary_label"]:
        label_names = ["0", "1"]
    elif args["labels"] is not None:
        label_names = args["labels"]

    is_copy = False if args["is_not_copy_png"] else True
    is_equalize = False if args["is_not_equalize"] else True

    if args["is_test_only"]:
        make_ts_dataset_train_test(
            args["input_dir"],
            args["output_dir"],
            label_names,
            l_size=args["l_size"],
            is_equalize=is_equalize,
            is_test_equalize=args["is_test_equalize"],
            is_copy=is_copy,
        )
    else:
        make_ts_dataset_train_val_test(
            args["input_dir"],
            args["output_dir"],
            label_names,
            l_size=args["l_size"],
            is_equalize=is_equalize,
            is_copy=is_copy,
        )
