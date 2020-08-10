#!/bin/bash
PWDDIR=`pwd`

PY_DIR=code

conda activate tfgpu

tf_base_class_all_data_2day_label() {
    # どこまでfreexeしたXceptionが一番loss下がるか調査
    python ${PY_DIR}/tf_base_class_all_data_2day_label.py -p "param_2day_label_trainable45.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label.py -p "param_2day_label_trainable65.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label.py -p "param_2day_label_trainable85.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label.py -p "param_2day_label_trainable105.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label.py -p "param_2day_label_trainable125.py" -m train
}

tf_base_class_all_data_2day_label_small() {
    # どこまでfreexeしたXceptionが一番loss下がるか調査
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small_trainable25.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small_trainable45.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small_trainable65.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small_trainable85.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small_trainable105.py" -m train
    python ${PY_DIR}/tf_base_class_all_data_2day_label_small.py -p "param_2day_label_small_trainable125.py" -m train
}

tf_base_class_all_data_positive_negative_line_label() {
    # 翌日が陽線、陰線か + 移動平均線つきについてモデル作成
    #python ${PY_DIR}/tf_base_class_all_data_positive_negative_line_label.py -p "param_positive_negative_line_label.py" -m train
    # predict
    python ${PY_DIR}/tf_base_class_all_data_positive_negative_line_label.py -p "param_positive_negative_line_label.py" -m predict
}

tf_base_class_all_data_positive_negative_line_label_small() {
    # 翌日が陽線、陰線か + 移動平均線つきについて、データ数減らしてパラメータチューニング
    python ${PY_DIR}/tf_base_class_all_data_positive_negative_line_label.py -m tuning -n_t 80
}

#tf_base_class_all_data_2day_label;
#tf_base_class_all_data_2day_label_small;

tf_base_class_all_data_positive_negative_line_label;
#tf_base_class_all_data_positive_negative_line_label_small;