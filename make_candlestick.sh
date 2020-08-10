#!/bin/bash
PWDDIR=`pwd`

PY_DIR=code

conda activate tfgpu

positive_negative_line_label() {
    IN_DIR=nikkei1000_split

    OUT_DIR=/d/work/candlestick_model/output/orig_image_all_positive_negative_line
    mkdir -p $OUT_DIR

    # 並列実行するためバックグラウンドプロセスで実行
    # 翌日が陽線、陰線か + 移動平均線つき
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/1-100.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/101-200.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/201-300.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/301-400.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/401-500.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/501-600.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/601-700.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/701-800.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/801-900.txt --is_mav_png --is_positive_negative_line_label &
    python ${PY_DIR}/make_candlestick.py -o $OUT_DIR -i $IN_DIR/901-994.txt --is_mav_png --is_positive_negative_line_label &
}

positive_negative_line_label;