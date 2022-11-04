import os
import sys
from typing import List

import pytest


sys.path.append(os.path.join(os.path.abspath("."), "ml_project/"))
sys.path.append("../")


def test_pred_train():
    exit_status_train = os.system("python train_pipeline.py")
    exit_status_predict = os.system("python predict_pipeline.py")
    assert exit_status_train == 1
    assert exit_status_predict == 1