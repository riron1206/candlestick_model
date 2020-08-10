#!/bin/bash
PWDDIR=`pwd`

PY_DIR=code

conda activate tfgpu

tf_base_class_all_data_positive_negative_line_label() {
    # 翌日が陽線、陰線か + 移動平均線つきについてモデル作成

    # スズキ
    #python ${PY_DIR}/tf_predict_best_model.py \
    #                -c 7269 \
    #                -d 2020-08-10 \
    #                -t_d 10 \
    #                -m /d/work/candlestick_model/output/model/ts_dataset_all_positive_negative_line_label/best_val_loss.h5 \
    #                -o /d/work/candlestick_model/output/model/ts_dataset_all_positive_negative_line_label/predict \
    #                --is_positive_negative_line_label

    # 日経225とか
    python ${PY_DIR}/tf_predict_best_model.py \
                    -t_d 40 \
                    -m /d/work/candlestick_model/output/model/ts_dataset_all_positive_negative_line_label/best_val_loss.h5 \
                    -o /d/work/candlestick_model/output/model/ts_dataset_all_positive_negative_line_label/predict \
                    --is_positive_negative_line_label
}

tf_base_class_all_data_positive_negative_line_label;