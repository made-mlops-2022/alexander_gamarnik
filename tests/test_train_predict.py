import os
import sys
# from typing import List

# import pytest


sys.path.append(os.path.join(os.path.abspath("."), "ml_project/"))
sys.path.append("../")


def test_pred_train():
    exit_status_train = os.system("python ml_project/train_pipeline.py configs/train_config.yaml")
    exit_status_predict = os.system("python ml_project/predict_pipeline.py configs/predict_config.yaml")
    assert exit_status_train == 0
    assert exit_status_predict == 0
