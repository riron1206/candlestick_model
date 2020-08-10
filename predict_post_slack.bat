@rem 作成日2020/08/10 シグナルcsvファイルを更新して銘柄コードをSlackに送る

call activate tfgpu

cd code

call python tf_predict_best_model.py --is_2day_label

call python tf_predict_best_model.py -m D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\best_val_loss.h5 -o D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\predict --is_positive_negative_line_label

call python post_my_slack.py

pause