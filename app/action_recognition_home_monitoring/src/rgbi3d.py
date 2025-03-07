import os

import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
import logging
from typing import Union, Tuple
from action_recognition_home_monitoring.src.utils import load_video, scale_video


USE_GPU = tf.config.list_physical_devices('GPU')


class RGBI3D:
    def __init__(self, model_keras_file: str):
        self.model_keras_file = model_keras_file
        self.model = keras.models.load_model(self.model_keras_file, safe_mode=False)
        _, self.n_frames, self.height, self.width, c = self.model.input.shape
        logging.info(f"Model input shape: {self.n_frames} frames with {self.height}x{self.width} resolution")

    def predict(self, video_path: Union[list[str], str], batch_size=8)-> Tuple[np.ndarray, np.ndarray]:
        """
        Receive video path, loads it, preprocess it and does inference with the model
        :param video_path: Path or list of paths to video/s
        :param batch_size: Batch size for doing predictions
        :return: Action recognition probabilities, Action classes IDs
        """
        if isinstance(video_path, str):
            video_path = [video_path]

        input_videos = []
        for v in video_path:
            video_preprocessed = scale_video(load_video(v))
            input_videos.append(video_preprocessed)

        input_for_models = np.concatenate(input_videos, axis=0)
        preds = self.model.predict(input_for_models, batch_size=batch_size, verbose=False)
        return preds

