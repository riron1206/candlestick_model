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


def filter_df_2day_label(
    csv=r"D:\work\candlestick_model\output\model\ts_dataset_all_2day_label\Xception\predict\pred.csv",
):
    th_price = 5000
    th_pb = 0.8

    df = pd.read_csv(csv, parse_dates=["date_exe"])

    # 最新日だけ
    df = df[df["date_exe"] == df["date_exe"].max()]

    for i, df_g in df.groupby(["date_exe"]):
        if df_g.shape[0] > 0:
            df_g = df_g[
                (df_g["pred_pb"] > th_pb)
                & (df_g["2day_pred_y"] > 0)
                & (df_g["date_exe_price"] > th_price)
            ]

    return df_g


def post_slack(name, text, post_url):
    requests.post(
        post_url,
        data=json.dumps({"text": text, "username": name, "icon_emoji": ":python:"}),
    )


if __name__ == "__main__":
    # webhookのエンドポイントのurl
    with open(
        os.path.join(
            r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\code\password",
            "slack.yml",
        )
    ) as f:
        config = yaml.load(f)
        post_url = config["post_url"]

    df = filter_df_2day_label()

    date = df.iloc[0]["date_exe"].date()
    # print(date)
    codes = df["code"].to_list()
    str_codes = map(str, codes)  # 格納される数値を文字列にする
    str_codes = ", ".join(str_codes)  # リストを文字列にする
    text = f"{str(date)} にシグナルが出た銘柄は {len(codes)} 件です。スコア大きい順に {str_codes} です。"
    # print(text)
    post_slack("2営業日後に上がりそうな株情報", text, post_url)
