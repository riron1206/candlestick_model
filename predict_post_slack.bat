@rem 作成日2020/08/10 シグナルcsvファイルを更新して銘柄コードをSlackに送る

call activate tfgpu

cd code

call python tf_predict_best_model.py --is_2day_label -t_d 2

call python post_my_slack.py