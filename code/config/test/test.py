import os, sys, argparse

sys.path.append(
    r"C:\Users\81908\jupyter_notebook\tf_2_work\stock_work\candlestick_model\code\config"
)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--param", type=str, default="param_2day_label.py")
    args = vars(ap.parse_args())

    if args["param"] == "param_2day_label.py":
        import param_2day_label as param_py
    else:
        import param_tmp as param_py

    params = param_py.get_class_fine_tuning_parameter_base()
    print(params)
