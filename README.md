# CodeSum
This repository contains code for the paper "Keyword-Guided Abstractive Code Summarization via Incorporating Structural and Contextual Information".
## About this code
The code is running with python3 and is developed for Tensorflow v1.15.2 and Keras v2.3.1.
## How to run?
1. To download the relevant data. The dataset we used is from the paper "Recommendations for Datasets for Source Code Summarization" where the relevant data can be downloaded [HERE](http://leclair.tech/data/funcom/)
2. To remove cloned code run creat_dup_new_data.py. The dup_data_result is the log for duplicate code, which is obtaind by duplicate code detector [HERE](https://github.com/microsoft/near-duplicate-code-detector)
3. To train a new model run train_dual_fc_info.py.
    `python3 train_dual_fc_info.py`
4. To obtain the prediction file run predict_dual_mul_dyn.py.
    `python3 predict_dual_mul_dyn.py`
