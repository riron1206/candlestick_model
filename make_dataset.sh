#!/bin/bash
PWDDIR=`pwd`

conda activate tfgpu

PY_DIR=code

positive_negative_line() {
    # 翌日が陽線、陰線か + 移動平均線つきの画像をtrain/testに分ける
    IN_DIR=/d/work/candlestick_model/output/orig_image_all_positive_negative_line

    OUT_DIR=/d/work/candlestick_model/output/ts_dataset_all_positive_negative_line_label
    mkdir -p ${OUT_DIR}

    # 全データで不均衡補正あり
    python ${PY_DIR}/make_dataset.py -i ${IN_DIR} -o ${OUT_DIR} -ls 0 1 2 3 --is_test_only
}

positive_negative_line_all() {
    # 不均衡補正なし
    # 翌日が陽線、陰線か + 移動平均線つきの画像をtrain/testに分ける
    IN_DIR=/d/work/candlestick_model/output/orig_image_all_positive_negative_line
    OUT_DIR=/d/work/candlestick_model/output/ts_dataset_all_positive_negative_line_label_all
    mkdir -p ${OUT_DIR}
    python ${PY_DIR}/make_dataset.py -i ${IN_DIR} -o ${OUT_DIR} -ls 0 1 2 3 --is_test_only --is_not_equalize
}

positive_negative_line_small() {
    # 画像枚数減らす
    # 翌日が陽線、陰線か + 移動平均線つきの画像をtrain/testに分ける
    IN_DIR=/d/work/candlestick_model/output/orig_image_all_positive_negative_line
    OUT_DIR=/d/work/candlestick_model/output/ts_dataset_all_positive_negative_line_label_small
    mkdir -p ${OUT_DIR}
    python ${PY_DIR}/make_dataset.py -i ${IN_DIR} -o ${OUT_DIR} -ls 0 1 2 3 --is_test_only -l_s 5000
}

#positive_negative_line;
positive_negative_line_all;
#positive_negative_line_small;