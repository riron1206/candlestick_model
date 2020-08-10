import os

# tensorflowのINFOレベルのログを出さないようにする
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras


def get_class_fine_tuning_parameter_base() -> dict:
    """
    Get parameter sample for class fine_tuning (like Keras)
    Returns:
        dict: parameter sample generated by trial object
    """
    my_IDG_options = {
        "rescale": 1.0 / 255.0,
        #'width_shift_range': 0.2,
        #'height_shift_range': 0.2,
        #'horizontal_flip': True,
        #'vertical_flip': True,
        #'shear_range': 20,
        #'zoom_range': 0.2,
        #'rotation_range': 20,
        #'channel_shift_range': 50,
        #'brightness_range': [0.3, 1.0],
        # "random_erasing_prob": 0.5,
        # "random_erasing_maxpixel": 255,
        #'mix_up_alpha': 0.2,
        #'random_crop': [224,224],
        #'ricap_beta': 0.3,
        #'ricap_use_same_random_value_on_batch': True,
        "randaugment_N": 3,
        "randaugment_M": 4,
        #'is_kuzushiji_gen': True,
        "cutmix_alpha": 1.0,
    }

    ## Augmentor使う場合のoption
    # train_augmentor_options = {
    #    'input_width': 80,
    #    'input_height': 80,
    #    'random_dist_prob': 0.3,
    #    'zoom_prob': 0.3,
    #    'zoom_min': 0.5
    #    , 'zoom_max': 1.9
    #    , 'flip_left_right': 0.3
    #    , 'flip_top_bottom': 0.3
    #    , 'random_erasing_prob': 0.3
    #    , 'random_erasing_area': 0.3
    # }

    return {
        "output_dir": r"D:\work\candlestick_model\output\model\ts_dataset_all_2day_label_small\Xception\_65",
        "gpu_count": 1,
        "img_rows": 80,
        "img_cols": 80,
        "channels": 3,
        "batch_size": 256,
        "classes": ["0", "1", "2"],
        "num_classes": 3,
        # "classes": ["0", "1"],
        # "num_classes": 2,
        # "train_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_2day_label_small\train",
        # "validation_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_2day_label_small\test",
        # "test_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_2day_label_small\test",
        "train_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_2day_label_small_class2_reduce\train",
        "validation_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_2day_label_small_class2_reduce\test",
        "test_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_2day_label_small_class2_reduce\test",
        "color_mode": "rgb",
        "class_mode": "categorical",  # generatorのラベルをone-hotベクトルに変換する場合。generatorのラベルを0か1のどちらかに変えるだけなら'binary'
        "activation": "softmax",
        # "loss": "categorical_crossentropy",
        "loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        "metrics": [
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
        "model_path": None,
        # "model_path": r"D:\work\candlestick_model\output\model\ts_dataset_all_2day_label\best_val_loss_20200727.h5",
        "num_epoch": 3,  # 200,
        "n_multitask": 1,  # マルチタスクのタスク数
        "multitask_pred_n_node": 1,  # マルチタスクの各クラス数
        # model param
        "weights": "imagenet",
        "choice_model": "Xception",
        # "choice_model": "model_paper",
        "fcpool": "GlobalAveragePooling2D",
        "is_skip_bn": False,
        # "trainable": "all",  # 249,
        "trainable": 65,
        "efficientnet_num": 3,
        # full layer param
        "fcs": [100],
        "drop": 0.3,
        "is_add_batchnorm": False,  # True,
        "l2_rate": 1e-4,
        # optimizer param
        "choice_optim": "sgd",
        "lr": 1e-1,
        "decay": 1e-5,
        "my_IDG_options": my_IDG_options,
        #'train_augmentor_options': train_augmentor_options,
        "TTA": "",  # 'flip',
        "TTA_rotate_deg": 0,
        "TTA_crop_num": 0,
        "TTA_crop_size": [224, 224],
        "preprocess": 1.0,
        "resize_size": [100, 100],
        "is_flow": False,
        "is_flow_from_directory": True,
        "is_flow_from_dataframe": False,
        # "is_lr_finder": False,
        "is_lr_finder": True,
        "is_class_weight": True,
    }
