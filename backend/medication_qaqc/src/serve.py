# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

import uvicorn
import io

from fastapi import FastAPI, Response
from PIL import Image

from ipex.utils.CustomVGG import CustomVGG
from ipex.training.TrainModel import TrainModel
from ipex.utils.logger import log

from model import PredictionPayload_IPEX, TrainPayload_IPEX

app = FastAPI()

@app.get("/ping")
async def ping():
    """Ping server to determine status

    Returns
    -------
    API response
        response from server on health status
    """
    return {"message": "Server is Running"}

@app.post("/train-ipex")
async def train(payload: TrainPayload_IPEX):

    vgg = CustomVGG()
    clf = TrainModel(vgg, payload.data_folder, payload.neg_class)
    clf.get_train_test_loaders(batch_size=5)
    log.info(f"Built train test loaders")
    epoch_loss, epoch_acc = clf.train(epochs=payload.epochs, learning_rate=payload.learning_rate, data_aug=payload.data_aug)
    log.info(f"Successfully Trained Model - Last Epoch Accuracy {epoch_acc} and Last Epoch Loss {epoch_loss}")
    accuracy, balanced_accuracy = clf.evaluate()
    log.info(f"Reporting Evaluation Models - Accuracy {accuracy} and Balanced Accuracy {balanced_accuracy}")
    clf.save_model(model_path=payload.modeldir)
    log.info(f"Successfully saved model to {payload.modeldir}")
    return {"msg": "Model trained successfully", "Model Location": payload.modeldir}

@app.post("/predict-ipex")
async def predict(payload: PredictionPayload_IPEX):

    y_pred_blind, file_list = cv_evaluator(trained_model=payload.trained_model_path, data_folder=payload.data_folder, batch_size=payload.batch_size)
    log.info(f'Prediction labels: {y_pred_blind} and associated files: {file_list}')
    return {"msg": "Model Inference Complete", "Prediction Output": list(zip(y_pred_blind, file_list))}

if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=5002, log_level="info")