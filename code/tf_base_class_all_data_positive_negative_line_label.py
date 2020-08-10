# -*- coding: utf-8 -*-
"""
tensorflow.kerasで分類モデル作成
ディレクトリやパラメータはget_class_fine_tuning_parameter_base()で変更する
Usage:
    # 指定ディレクトリについてgenerator作ってモデル学習
    $ python tf_base_class_all_data_positive_negative_line_label.py -m train
    ※最適な学習率確認する場合はget_class_fine_tuning_parameter_base()のis_lr_finderをtrueにする

    # 指定ディレクトリについてモデル予測
    $ python tf_base_class_all_data_positive_negative_line_label.py -m predict

    # optunaでパラメータチューニング
    # 変更するパラメータは Objective.get_class_fine_tuning_parameter_suggestions() で変更する
    $ python tf_base_class_all_data_positive_negative_line_label.py -m tuning -n_t 50 -t_out_dir D:\work\candlestick_model\output\model\ts_dataset_all_2day_label\optuna
"""
import os
import sys
import time
import shutil
import argparse
import traceback
import pathlib

# tensorflowのINFOレベルのログを出さないようにする
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml
from tqdm import tqdm
import numpy as np
import optuna
import tensorflow as tf
from tensorflow import keras

keras_py_path = r"C:\Users\81908\jupyter_notebook\tfgpu_py36_work\02_keras_py"
sys.path.append(keras_py_path)
from dataset import plot_log, util
from model import tf_define_model as define_model
from model import tf_my_callback as my_callback
from model import tf_lr_finder as lr_finder
from model import tf_pooling as pooling
from model import my_class_weight
from transformer import tf_my_generator as my_generator
from transformer import tf_get_train_valid_test as get_train_valid_test
from predicter import tf_grad_cam as grad_cam
from predicter import tf_base_predict as base_predict
from predicter import roc_curve, conf_matrix, ensemble_predict

import model_paper


def train_directory(args):
    """指定ディレクトリについてgenerator作ってモデル学習"""
    print("train_directory")
    # ### train validation data load ### #
    d_cls = get_train_valid_test.LabeledDataset(
        [args["img_rows"], args["img_cols"], args["channels"]],
        args["batch_size"],
        valid_batch_size=args["batch_size"],
        train_samples=len(util.find_img_files(args["train_data_dir"])),
        valid_samples=len(util.find_img_files(args["validation_data_dir"])),
    )
    if args["is_flow"]:
        # 指定ディレクトリの前処理済み画像、ラベル、ファイルパスロード
        d_cls.X_train, d_cls.y_train, train_paths = base_dataset.load_my_data(
            args["train_data_dir"],
            classes=args["classes"],
            img_height=args["img_rows"],
            img_width=args["img_cols"],
            channel=args["channels"],
            is_pytorch=False,
        )
        d_cls.X_valid, d_cls.y_valid, valid_paths = base_dataset.load_my_data(
            args["validation_data_dir"],
            classes=args["classes"],
            img_height=args["img_rows"],
            img_width=args["img_cols"],
            channel=args["channels"],
            is_pytorch=False,
        )
        d_cls.X_train, d_cls.X_valid = d_cls.X_train * 255.0, d_cls.X_valid * 255.0
        d_cls.create_my_generator_flow(my_IDG_options=args["my_IDG_options"])

    elif args["is_flow_from_directory"]:
        d_cls.create_my_generator_flow_from_directory(
            args["train_data_dir"],
            args["classes"],
            valid_data_dir=args["validation_data_dir"],
            color_mode=args["color_mode"],
            class_mode=args["class_mode"],
            my_IDG_options=args["my_IDG_options"],
        )
        # d_cls.train_gen_augmentor = d_cls.create_augmentor_util_from_directory(args['train_data_dir']
        #                                                                       , args['batch_size']
        #                                                                       , augmentor_options=args['train_augmentor_options'])

    # binaryラベルのgeneratorをマルチタスクgeneratorに変換するラッパー
    if args["n_multitask"] > 1 and args["multitask_pred_n_node"] == 1:
        d_cls.train_gen = get_train_valid_test.binary_generator_multi_output_wrapper(
            d_cls.train_gen
        )
        d_cls.valid_gen = get_train_valid_test.binary_generator_multi_output_wrapper(
            d_cls.valid_gen
        )

    # ### model ### #
    if args["model_path"] is not None:
        print(f"INFO: load model: {args['model_path']}")
        model = keras.models.load_model(args["model_path"], compile=False)
    else:
        os.makedirs(args["output_dir"], exist_ok=True)
        if args["choice_model"] == "model_paper":
            model = model_paper.create_paper_cnn(
                input_shape=(args["img_cols"], args["img_rows"], args["channels"]),
                num_classes=args["num_classes"],
                activation=args["activation"],
            )
        else:
            model, orig_model = define_model.get_fine_tuning_model(
                args["output_dir"],
                args["img_rows"],
                args["img_cols"],
                args["channels"],
                args["num_classes"],
                args["choice_model"],
                trainable=args["trainable"],
                fcpool=args["fcpool"],
                fcs=args["fcs"],
                drop=args["drop"],
                activation=args["activation"],
                weights=args["weights"],
            )
    optim = define_model.get_optimizers(
        choice_optim=args["choice_optim"], lr=args["lr"], decay=args["decay"]
    )
    model.compile(loss=args["loss"], optimizer=optim, metrics=args["metrics"])

    cb = my_callback.get_base_cb(
        args["output_dir"],
        args["num_epoch"],
        early_stopping=args["num_epoch"] // 4,
        monitor="val_" + args["metrics"][0],
        metric=args["metrics"][0],
    )

    # lr_finder
    if args["is_lr_finder"] == True:
        # 最適な学習率確認して関数抜ける
        lr_finder.run(
            model,
            d_cls.train_gen,
            args["batch_size"],
            d_cls.init_train_steps_per_epoch,
            output_dir=args["output_dir"],
        )
        return

    # 各クラスのlossを加重平均にする場合
    class_weight = {c: 1.0 for c in args["classes"]}
    if args["is_class_weight"]:
        class_weight = my_class_weight.cal_weight(
            args["classes"], args["train_data_dir"]
        )

    # ### train ### #
    start_time = time.time()
    hist = model.fit(
        d_cls.train_gen,
        steps_per_epoch=d_cls.init_train_steps_per_epoch,
        epochs=args["num_epoch"],
        validation_data=d_cls.valid_gen,
        validation_steps=d_cls.init_valid_steps_per_epoch,
        verbose=1,  # 1:ログをプログレスバーで標準出力 2:最低限の情報のみ出す
        callbacks=cb,
        class_weight=class_weight,
    )
    end_time = time.time()
    print("Elapsed Time : {:.2f}sec".format(end_time - start_time))

    model.save(os.path.join(args["output_dir"], "model_last_epoch.h5"))

    plot_log.plot_results(
        args["output_dir"],
        os.path.join(args["output_dir"], "tsv_logger.tsv"),
        acc_metric=args["metrics"][0],
    )

    return hist


def pred_directory(args):
    """指定ディレクトリについてモデル予測"""
    # ### test data load ### #
    d_cls = get_train_valid_test.LabeledDataset(
        [args["img_rows"], args["img_cols"], args["channels"]],
        args["batch_size"],
        valid_batch_size=args["batch_size"],
    )
    if args["is_flow"]:
        # 指定ディレクトリの前処理済み画像、ラベル、ファイルパスロード
        d_cls.X_test, d_cls.y_test, test_paths = base_dataset.load_my_data(
            args["test_data_dir"],
            classes=args["classes"],
            img_height=args["img_rows"],
            img_width=args["img_cols"],
            channel=args["channels"],
            is_pytorch=False,
        )
        d_cls.create_test_generator()

    elif args["is_flow_from_directory"]:
        d_cls.create_my_generator_flow_from_directory(
            args["train_data_dir"],
            args["classes"],
            test_data_dir=args["test_data_dir"],
            color_mode=args["color_mode"],
            class_mode=args["class_mode"],
            my_IDG_options={"rescale": 1 / 255.0},
        )

    # binaryラベルのgeneratorをマルチタスクgeneratorに変換するラッパー
    if args["n_multitask"] > 1 and args["multitask_pred_n_node"] == 1:
        d_cls.test_gen = get_train_valid_test.binary_generator_multi_output_wrapper(
            d_cls.test_gen
        )

    # generator predict TTA
    # load_model = keras.models.load_model(os.path.join(args['output_dir'], 'best_val_loss.h5'))
    load_model = keras.models.load_model(
        os.path.join(args["output_dir"], "best_val_accuracy.h5")
    )
    pred_tta = base_predict.predict_tta_generator(
        load_model,
        d_cls.test_gen,
        TTA=args["TTA"],
        TTA_rotate_deg=args["TTA_rotate_deg"],
        TTA_crop_num=args["TTA_crop_num"],
        TTA_crop_size=args["TTA_crop_size"],
        resize_size=[args["img_rows"], args["img_cols"]],
    )
    pred_tta_df = base_predict.get_predict_generator_results(
        pred_tta, d_cls.test_gen, classes_list=args["classes"]
    )
    # 混同行列作成
    base_predict.conf_matrix_from_pred_classes_generator(
        pred_tta_df, args["classes"], args["output_dir"]
    )


class OptunaCallback(keras.callbacks.Callback):
    """
    Optunaでの枝刈り（最終的な結果がどのぐらいうまくいきそうかを大まかに予測し、良い結果を残すことが見込まれない試行は、最後まで行うことなく早期終了）
    https://qiita.com/koshian2/items/107c386f81c9bb7f8df3
    """

    def __init__(self, trial, prune):
        self.trial = trial
        self.prune = prune

    def on_epoch_end(self, epoch, logs):
        current_val_error = logs["val_loss"]  # 1.0 - logs["val_accuracy"]
        # epochごとの値記録（intermediate_values）
        self.trial.report(current_val_error, step=epoch)
        if self.prune == True:
            # 打ち切り判定
            if self.trial.should_prune(epoch):
                # MedianPrunerのデフォルトの設定で、最初の5trialをたたき台して使って、以降のtrialで打ち切っていく
                # raise optuna.structs.TrialPruned()
                raise optuna.TrialPruned()


class Objective(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.trial_best_loss = 1000.0
        self.trial_best_err = 1000.0

    def get_class_fine_tuning_parameter_suggestions(self, trial) -> dict:
        """
        Get parameter sample for class fine_tuning (like Keras)
        Args:
            trial(trial.Trial):
        Returns:
            dict: parameter sample generated by trial object
        """

        my_IDG_options = {
            "rescale": 1.0 / 255.0,
            #'width_shift_range': trial.suggest_categorical('height_shift_range', [0.0, 0.25]),
            #'height_shift_range': trial.suggest_categorical('height_shift_range', [0.0, 0.25]),
            #'horizontal_flip': trial.suggest_categorical('horizontal_flip', [True, False]),
            #'vertical_flip': trial.suggest_categorical('vertical_flip', [True, False]),
            #'shear_range': trial.suggest_categorical('shear_range', [0.0, 20, 50]),
            # "zoom_range": trial.suggest_categorical("zoom_range", [0.0, 0.2, 0.5]),
            #'rotation_range': trial.suggest_categorical('rotation_range', [0.0, 45, 60, 90]),
            #'channel_shift_range': trial.suggest_categorical('channel_shift_range', [0.0, 100, 200]),
            #'brightness_range': trial.suggest_categorical('brightness_range', [[1.0, 1.0], [0.3, 1.0]]),
            # MyImageDataGenerator param
            # "random_erasing_prob": trial.suggest_categorical("random_erasing_prob", [0.0, 0.5]),
            # "random_erasing_maxpixel": 255.0,
            #'mix_up_alpha': trial.suggest_categorical('mix_up_alpha', [0.0, 0.2]),
            #'ricap_beta': trial.suggest_categorical('ricap_beta', [0.0, 0.3]),
            #'is_kuzushiji_gen': trial.suggest_categorical('is_kuzushiji_gen', [False]),
            # "grayscale_prob": trial.suggest_categorical("grayscale_prob", [0.0, 0.3]),
            "cutmix_alpha": trial.suggest_categorical("cutmix_alpha", [0.0, 0.5, 1.0]),
            # "randaugment_N": trial.suggest_categorical("randaugment_N", [0, 3]),
            # "randaugment_M": trial.suggest_categorical("randaugment_M", [0, 4]),
            # "randaugment_N": 3,
            # "randaugment_M": 4,
        }
        is_randaugment = trial.suggest_categorical("randaugment", [True, False])
        if is_randaugment:
            my_IDG_options["randaugment_N"] = 3
            my_IDG_options["randaugment_M"] = 4

        ## Augmentor使う場合のoption
        # train_augmentor_options = {
        #    'rescale': 1.0/255.0,
        #    'rotate90': trial.suggest_categorical('rotate90', [0.0, 0.5]),
        #    'rotate180': trial.suggest_categorical('rotate180', [0.0, 0.5]),
        #    'rotate270': trial.suggest_categorical('rotate270', [0.0, 0.5]),
        #    'rotate_prob': trial.suggest_categorical('rotate_prob', [0.0, 0.5]),
        #    'rotate_max_left': trial.suggest_categorical('rotate_max_left', [20, 60, 90]),
        #    'rotate_max_right': trial.suggest_categorical('rotate_max_right', [20, 60, 90]),
        #    'crop_prob': trial.suggest_categorical('crop_prob', [0.0, 0.5]),
        #    'crop_area': trial.suggest_categorical('crop_area', [0.8, 0.5]),
        #    'crop_by_size_prob': trial.suggest_categorical('crop_by_size_prob', [0.0, 0.5]),
        #    'crop_by_width': trial.suggest_categorical('crop_by_width', [224]),
        #    'crop_by_height': trial.suggest_categorical('crop_by_height', [224]),
        #    'crop_by_centre': trial.suggest_categorical('crop_by_centre', [True, False]),
        #    'shear_prob': trial.suggest_categorical('shear_prob', [0.0, 0.5]),
        #    'shear_magni': trial.suggest_categorical('shear_magni', [20, 50]),
        #    'skew_prob': trial.suggest_categorical('skew_prob', [0.0, 0.5]),
        #    'skew_magni': trial.suggest_categorical('skew_magni', [20, 50]),
        #    'zoom_prob': trial.suggest_categorical('zoom_prob', [0.0, 0.5]),
        #    'zoom_min': trial.suggest_categorical('zoom_min', [0.2, 0.5, 0.9]),
        #    'zoom_max': trial.suggest_categorical('zoom_max', [1.2, 1.5, 1.9]),
        #    'flip_left_right': trial.suggest_categorical('flip_left_right', [0.0, 0.5]),
        #    'flip_top_bottom': trial.suggest_categorical('flip_top_bottom', [0.0, 0.5]),
        #    'random_erasing_prob': trial.suggest_categorical('random_erasing_prob', [0.0, 0.5]),
        #    'random_erasing_area': trial.suggest_categorical('random_erasing_area', [0.3]),
        #    'random_dist_prob': trial.suggest_categorical('random_dist_prob', [0.0, 0.5]),
        #    'random_dist_grid_width': trial.suggest_categorical('random_dist_grid_width', [4]),
        #    'random_dist_grid_height': trial.suggest_categorical('random_dist_grid_height', [4]),
        #    'random_dist_grid_height': trial.suggest_categorical('random_dist_grid_height', [4]),
        #    'random_dist_magnitude': trial.suggest_categorical('random_dist_magnitude', [8]),
        #    'black_and_white': trial.suggest_categorical('black_and_white', [0.0, 0.5]),
        #    'greyscale': trial.suggest_categorical('greyscale', [0.0, 0.5]),
        #    'invert': trial.suggest_categorical('invert', [0.0, 0.5])
        # }

        return {
            "output_dir": r"D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\optuna_small",
            "gpu_count": 1,
            "img_rows": 80,
            "img_cols": 80,
            "channels": 3,
            "batch_size": 256,
            "classes": ["0", "1", "2", "3"],
            "num_classes": 4,
            # "classes": ["0", "1"],
            # "num_classes": 2,
            "train_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_positive_negative_line_label_small\train",
            "validation_data_dir": r"D:\work\candlestick_model\output\ts_dataset_all_positive_negative_line_label_small\test",
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
            "num_epoch": 50,
            "n_multitask": 1,  # マルチタスクのタスク数
            "multitask_pred_n_node": 1,  # マルチタスクの各クラス数
            # model param
            "weights": "imagenet",
            # "choice_model": "Xception",
            "choice_model": trial.suggest_categorical(
                "choice_model",
                ["VGG16", "Xception", "InceptionV3", "model_paper"],  # , "ResNet50"]
            ),
            "fcpool": "GlobalAveragePooling2D",
            "is_skip_bn": False,
            "trainable": "all",
            "efficientnet_num": 3,
            # full layer param
            # "fcs": [512, 256],
            "fcs": trial.suggest_categorical(
                "fcs", [[], [100]]  # , [256], [512, 256], [1024, 512, 256]
            ),
            "drop": 0.3,  # 0.0はドロップなし
            "is_add_batchnorm": False,
            "l2_rate": 1e-4,
            # optimizer param
            # "choice_optim": "sgd",
            "choice_optim": trial.suggest_categorical("choice_optim", ["sgd", "adam"]),
            # "lr": 1e-2,
            "lr": trial.suggest_categorical("lr", [1e-1]),
            "decay": 1e-5,
            # data augment
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
            "is_class_weight": True,
        }

    def trial_train_directory(self, trial, args):
        keras.backend.clear_session()
        # ### train validation data load ### #
        d_cls = get_train_valid_test.LabeledDataset(
            [args["img_rows"], args["img_cols"], args["channels"]],
            args["batch_size"],
            valid_batch_size=args["batch_size"],
            train_samples=len(util.find_img_files(args["train_data_dir"])),
            valid_samples=len(util.find_img_files(args["validation_data_dir"])),
        )
        if args["is_flow"]:
            # 指定ディレクトリの前処理済み画像、ラベル、ファイルパスロード
            d_cls.X_train, d_cls.y_train, train_paths = base_dataset.load_my_data(
                args["train_data_dir"],
                classes=args["classes"],
                img_height=args["img_rows"],
                img_width=args["img_cols"],
                channel=args["channels"],
                is_pytorch=False,
            )
            d_cls.X_valid, d_cls.y_valid, valid_paths = base_dataset.load_my_data(
                args["validation_data_dir"],
                classes=args["classes"],
                img_height=args["img_rows"],
                img_width=args["img_cols"],
                channel=args["channels"],
                is_pytorch=False,
            )
            d_cls.X_train, d_cls.X_valid = d_cls.X_train * 255.0, d_cls.X_valid * 255.0
            d_cls.create_my_generator_flow(my_IDG_options=args["my_IDG_options"])

        elif args["is_flow_from_directory"]:
            d_cls.create_my_generator_flow_from_directory(
                args["train_data_dir"],
                args["classes"],
                valid_data_dir=args["validation_data_dir"],
                color_mode=args["color_mode"],
                class_mode=args["class_mode"],
                my_IDG_options=args["my_IDG_options"],
            )
            # d_cls.train_gen_augmentor = d_cls.create_augmentor_util_from_directory(args['train_data_dir']
            #                                                                       , args['batch_size']
            #                                                                       , augmentor_options=args['train_augmentor_options'])

        # binaryラベルのgeneratorをマルチタスクgeneratorに変換するラッパー
        if args["n_multitask"] > 1 and args["multitask_pred_n_node"] == 1:
            d_cls.train_gen = get_train_valid_test.binary_generator_multi_output_wrapper(
                d_cls.train_gen
            )
            d_cls.valid_gen = get_train_valid_test.binary_generator_multi_output_wrapper(
                d_cls.valid_gen
            )

        # ### model ### #
        if args["model_path"] is not None:
            model = keras.models.load_model(args["model_path"], compile=False)
        else:
            os.makedirs(args["output_dir"], exist_ok=True)
            if args["choice_model"] == "model_paper":
                model = model_paper.create_paper_cnn(
                    input_shape=(args["img_cols"], args["img_rows"], args["channels"]),
                    num_classes=args["num_classes"],
                    activation=args["activation"],
                )
            else:
                model, orig_model = define_model.get_fine_tuning_model(
                    args["output_dir"],
                    args["img_rows"],
                    args["img_cols"],
                    args["channels"],
                    args["num_classes"],
                    args["choice_model"],
                    trainable=args["trainable"],
                    fcpool=args["fcpool"],
                    fcs=args["fcs"],
                    drop=args["drop"],
                    activation=args["activation"],
                    weights=args["weights"],
                )
        optim = define_model.get_optimizers(
            choice_optim=args["choice_optim"], lr=args["lr"], decay=args["decay"]
        )
        model.compile(loss=args["loss"], optimizer=optim, metrics=args["metrics"])

        cb = my_callback.get_base_cb(
            args["output_dir"],
            args["num_epoch"],
            early_stopping=20,
            monitor="val_" + args["metrics"][0],
            metric=args["metrics"][0],
        )  # args['num_epoch']//3
        cb.append(OptunaCallback(trial, True))

        # 各クラスのlossを加重平均にする場合
        class_weight = {c: 1.0 for c in args["classes"]}
        if args["is_class_weight"]:
            class_weight = my_class_weight.cal_weight(
                args["classes"], args["train_data_dir"]
            )

        # ### train ### #
        hist = model.fit(
            d_cls.train_gen,
            steps_per_epoch=d_cls.init_train_steps_per_epoch,
            epochs=args["num_epoch"],
            validation_data=d_cls.valid_gen,
            validation_steps=d_cls.init_valid_steps_per_epoch,
            verbose=2,  # 1:ログをプログレスバーで標準出力 2:最低限の情報のみ出す
            callbacks=cb,
            class_weight=class_weight,
        )

        return hist

    def __call__(self, trial):
        args = self.get_class_fine_tuning_parameter_suggestions(trial)
        print(args)
        # optuna v0.18以上だとtryで囲まないとエラーでtrial落ちる
        try:
            # train
            hist = self.trial_train_directory(trial, args)

            check_loss = np.min(hist.history["val_loss"])  # check_dataは小さい方が精度良いようにしておく
            if check_loss < self.trial_best_loss:
                print(
                    "check_loss, trial_best_loss:",
                    str(check_loss),
                    str(self.trial_best_loss),
                )
                self.trial_best_loss = check_loss
                if (
                    os.path.exists(os.path.join(args["output_dir"], "best_val_loss.h5"))
                    == True
                ):
                    shutil.copyfile(
                        os.path.join(args["output_dir"], "best_val_loss.h5"),
                        os.path.join(args["output_dir"], "best_trial_loss.h5"),
                    )

            check_err = 1.0 - np.max(
                hist.history["val_accuracy"]
            )  # check_dataは小さい方が精度良いようにしておく
            if check_err < self.trial_best_err:
                print(
                    "check_err, trial_best_err:",
                    str(check_err),
                    str(self.trial_best_err),
                )
                self.trial_best_err = check_err
                if (
                    os.path.exists(
                        os.path.join(args["output_dir"], "best_val_accuracy.h5")
                    )
                    == True
                ):
                    shutil.copyfile(
                        os.path.join(args["output_dir"], "best_val_accuracy.h5"),
                        os.path.join(args["output_dir"], "best_trial_accuracy.h5"),
                    )

            # acc とloss の記録
            trial.set_user_attr("loss", np.min(hist.history["loss"]))
            trial.set_user_attr("val_loss", np.min(hist.history["val_loss"]))
            trial.set_user_attr("val_accuracy", str(max(hist.history["val_accuracy"])))
            trial.set_user_attr("val_recall", str(max(hist.history["val_recall"])))
            trial.set_user_attr(
                "val_precision", str(max(hist.history["val_precision"]))
            )
            trial.set_user_attr("val_auc", str(max(hist.history["val_auc"])))

            return np.min(hist.history["val_loss"])
        except Exception as e:
            traceback.print_exc()  # Exceptionが発生した際に表示される全スタックトレース表示
            return e  # 例外を返さないとstudy.csvにエラー内容が記載されない


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["train", "predict", "tuning"])
    parser.add_argument("--grad_cam_model_path", type=str, default=None)
    parser.add_argument("--grad_cam_image_dir", type=str, default=None)
    parser.add_argument(
        "--study_name", help="Optuna trials study name", type=str, default="study"
    )
    parser.add_argument(
        "-n_t", "--n_trials", help="Optuna trials number", type=int, default=2
    )
    parser.add_argument(
        "-t_out_dir",
        "--tuning_output_dir",
        help="Optuna trials output_dir",
        type=str,
        default=r"D:\work\candlestick_model\output\model\ts_dataset_all_positive_negative_line_label\optuna",
    )
    parser.add_argument(
        "-p", "--param", help="param config py", type=str, default="param_2day_label.py"
    )
    p_args = parser.parse_args()

    return p_args


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")

    p_args = get_args()

    # config置き場
    sys.path.append(
        r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\code\config"
    )
    # パラメータ.py import
    if p_args.param == "param_positive_negative_line_label.py":
        import param_positive_negative_line_label as param_py

    if p_args.mode == "train":
        args = param_py.get_class_fine_tuning_parameter_base()
        train_directory(args)

    if p_args.mode == "predict":
        args = param_py.get_class_fine_tuning_parameter_base()
        pred_directory(args)

    if p_args.grad_cam_model_path is not None and p_args.grad_cam_image_dir is not None:
        args = param_py.get_class_fine_tuning_parameter_base()
        for i, p in tqdm(enumerate(util.find_img_files(p_args.grad_cam_image_dir))):
            # 50枚ごとにモデル再ロード
            if i % 50 == 0:
                keras.backend.clear_session()
                keras.backend.set_learning_phase(0)
                model = keras.models.load_model(
                    p_args.grad_cam_model_path, compile=False
                )
            # p_args.grad_cam_image_dirと同じディレクトリにGradCAM画像出力
            grad_cam.image2gradcam(model, p, is_gradcam_plus=False)

    if p_args.mode == "tuning":
        os.makedirs(p_args.tuning_output_dir, exist_ok=True)
        study = optuna.create_study(
            direction="minimize",
            study_name=p_args.study_name,
            storage=f"sqlite:///{p_args.tuning_output_dir}/{p_args.study_name}.db",
            load_if_exists=True,
        )
        study.optimize(
            Objective(p_args.tuning_output_dir),
            n_trials=p_args.n_trials,
            # timeout=60 * 30,  # 全trial30分
        )
        study.trials_dataframe().to_csv(
            f"{p_args.tuning_output_dir}/{p_args.study_name}_history.csv", index=False
        )
        print(f"\nstudy.best_params:\n{study.best_params}")
        print(f"\nstudy.best_trial:\n{study.best_trial}")
