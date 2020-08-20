#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自分のSlackに結果メッセージを飛ばす
参考:
https://vaaaaaanquish.hatenablog.com/entry/2017/09/27/154210
https://qiita.com/ik-fib/items/b4a502d173a22b3947a0

Usage:
    $ python post_my_slack.py
"""
import os
import requests
import json
import pandas as pd
import yaml
import warnings

warnings.filterwarnings("ignore")


def post_slack(name, text):
    """slackに情報投げる"""
    # webhookのエンドポイントのurl
    with open(
        os.path.join(
            r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\code\password",
            "slack.yml",
        )
    ) as f:
        config = yaml.load(f)
        post_url = config["post_url"]
    requests.post(
        post_url,
        data=json.dumps({"text": text, "username": name, "icon_emoji": ":python:"}),
    )


def post_filter_df_2day_label(
    csv=r"D:\work\candlestick_model\output\model\ts_dataset_all_2day_label\Xception\predict\pred.csv",
    th_price=5000,
    th_pb=0.8,
):
    """2営業日後に上がりそうな株情報絞ってslackに投げる"""
    df = pd.read_csv(csv, parse_dates=["date_exe"])

    # 最新日だけ
    df = df[df["date_exe"] == df["date_exe"].max()]

    # 条件に合う行だけにする
    df_result = None
    for i, df_g in df.groupby(["date_exe"]):
        if df_g.shape[0] > 0:
            df_g = df_g[
                (df_g["pred_pb"] > th_pb)  # スコア高いのだけに
                & (df_g["2day_pred_y"] > 0)  # ラベル1,2だけに
                & (df_g["date_exe_price"] < th_price)  # 価格が高すぎるのは除く
            ]
            df_g = df_g.sort_values(by="pred_pb", ascending=False)  # スコアの降順にする
            df_g = df_g.head(10) if df_g.shape[0] > 10 else df_g  # 10件に絞る
            df_result = df_g
    # print(df_result)

    # 株コードslackに投げる
    date = df_result.iloc[0]["date_exe"].date()
    # print(date)
    codes = df_result["code"].to_list()
    # str_codes = map(str, codes)  # 格納される数値を文字列にする
    # str_codes = ", ".join(str_codes)  # リストを文字列にする
    text = f"{str(date)} にシグナル出た銘柄上位 {len(codes)} 件"  # {str_codes}"
    for code in codes:
        text = (
            text
            + "\n"
            + f"https://stocks.finance.yahoo.co.jp/stocks/chart/?code={str(code)}.T"
        )
    print(text)
    post_slack("2営業日後に上がりそうな株情報", text)


def post_filter_df_positive_negative_line_label(
    csv=r"D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\predict\pred.csv",
    th_price=5000,
    th_pb=0.45,
):
    """翌日が陽線になりそうな株情報絞ってslackに投げる"""
    df = pd.read_csv(csv, parse_dates=["date_exe"])

    # 最新日だけ
    df = df[df["date_exe"] == df["date_exe"].max()]

    # 条件に合う行だけにする
    df_result = None
    for i, df_g in df.groupby(["date_exe"]):
        if df_g.shape[0] > 0:
            df_g = df_g[
                (df_g["pred_pb"] > th_pb)  # スコア高いのだけに
                & (df_g["next_day_pred_y"] > 1)  # ラベル2,3だけに
                & (df_g["date_exe_price"] < th_price)  # 価格が高すぎるのは除く
            ]
            df_g = df_g.sort_values(by="pred_pb", ascending=False)  # スコアの降順にする
            df_g = df_g.head(10) if df_g.shape[0] > 10 else df_g  # 10件に絞る
            df_result = df_g
    # print(df_result)

    # 株コードslackに投げる
    date = df_result.iloc[0]["date_exe"].date()
    # print(date)
    codes = df_result["code"].to_list()
    # str_codes = map(str, codes)  # 格納される数値を文字列にする
    # str_codes = ", ".join(str_codes)  # リストを文字列にする
    text = f"{str(date)} にシグナル出た銘柄上位 {len(codes)} 件"  # {str_codes}"
    for code in codes:
        text = (
            text
            + "\n"
            + f"https://stocks.finance.yahoo.co.jp/stocks/chart/?code={str(code)}.T"
        )
    print(text)
    post_slack("翌日が陽線になりそうな株情報", text)


if __name__ == "__main__":

    # ###################### 翌日が陽線か ######################
    post_filter_df_positive_negative_line_label()
    ############################################################

    # ################### 2営業日後に上がりそうか ##############
    post_filter_df_2day_label()
    ############################################################
