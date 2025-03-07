import os

import numpy as np

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from typing import Union, Tuple
from action_recognition_home_monitoring.src.utils import load_video, scale_video
from action_recognition_home_monitoring.src.flowestimator import FlowEstimatorTVL1CPU, FlowEstimatorTVL1CUDA


USE_GPU = tf.config.list_physical_devices('GPU')


class TwoStream:
    def __init__(self, rgb_model_keras_file: str, flow_model_keras_file: str):
        self.rgb_model_keras_file = rgb_model_keras_file
        self.flow_model_keras_file = flow_model_keras_file
        self.rgb_model = keras.models.load_model(self.rgb_model_keras_file, safe_mode=False)
        self.rgb_model = keras.models.Model(self.rgb_model.input,
                                            self.rgb_model.get_layer("logits").output,
                                            name=self.rgb_model.name)
        self.flow_model = keras.models.load_model(self.flow_model_keras_file, safe_mode=False)
        self.flow_model = keras.models.Model(self.flow_model.input,
                                             self.flow_model.get_layer("logits").output,
                                             name=self.flow_model.name)

        rgb_input = keras.layers.Input(self.rgb_model.input.shape[1:], name="rgb_input")
        flow_input = keras.layers.Input(self.flow_model.input.shape[1:], name="flow_input")

        rgb_stream = self.rgb_model(rgb_input)
        flow_stream = self.flow_model(flow_input)

        logits = rgb_stream + flow_stream

        preds = keras.layers.Activation("softmax", name="softmax")(logits)

        self.two_stream_model = keras.models.Model([rgb_input, flow_input], preds, name="TwoStream_I3D")

        if USE_GPU:
            self.flow_estimator = FlowEstimatorTVL1CUDA(reduce_factor=1)
        else:
            self.flow_estimator = FlowEstimatorTVL1CPU(reduce_factor=1)

    def predict(self, video_path: Union[list[str], str], batch_size=8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Receive video path, loads it, preprocess it and does inference with the model
        :param video_path: Path or list of paths to video/s
        :param batch_size: Batch size
        :return: Action recognition probabilities, Action classes IDs
        """
        if isinstance(video_path, str):
            video_path = [video_path]

        input_videos = []
        input_flows = []
        for v in video_path:
            video_preprocessed = scale_video(load_video(v))
            input_videos.append(video_preprocessed)
            flow_preprocessed = self.flow_estimator.computeFlow(video_preprocessed[0])
            input_flows.append(flow_preprocessed)

        inputs_rgb = np.concatenate(input_videos, axis=0)
        inputs_flows = np.concatenate(input_flows, axis=0)
        preds = self.two_stream_model.predict([inputs_rgb, inputs_flows], batch_size=batch_size, verbose=False)
        return preds
