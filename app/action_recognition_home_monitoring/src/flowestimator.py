import cv2
from abc import ABC, abstractmethod
import numpy as np
import time


class FlowEstimator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def computeFlow(self, video):
        pass

    def estimate(self, video):
        """
        Estimate Optical Flow over the whole video
        :param video: Numpy of N frames
        :return: Normalized estimated flow, time needed to estimate Flow
        """
        start_time = time.time()
        flow_estimated = self.computeFlow(video)
        end_time = time.time()
        time_spent = end_time - start_time
        return flow_estimated, time_spent


class FlowEstimatorTVL1(FlowEstimator, ABC):

    def __init__(self, skip_frames=0, reduce_factor=0.5, concatenate=True):

        self.skip_frames = skip_frames
        self.reduce_factor = reduce_factor
        self.concatenate = concatenate

        super(FlowEstimatorTVL1, self).__init__()

    @abstractmethod
    def computeTVL1(self, previous, next, bound=20):
        pass

    def _concatFrames(self, frames_list: list):
        """
        Concatenate frames to be estimated jointly
        :param frames_list:
        :return: Two vertical frames to estimate the flow of the whole video
        """
        # Vertical concatenation of the frames from the first one
        prev_frame = cv2.vconcat(frames_list[:-1])

        # Vertical concatenation of the frames from the second one
        next_frame = cv2.vconcat(frames_list[1:])

        return prev_frame, next_frame

    def _preprocess(self, video):
        video = video[0::(self.skip_frames + 1)]
        video_list = []
        for frame in video:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, None, fx=self.reduce_factor, fy=self.reduce_factor)
            video_list.append(frame)
        return video_list

    def _reconstructEstimated(self, flow, frame_height):
        """
        Reconstruct separated frames from a concatenated estimated Optical Flow
        :param flow: Estimated optical flow
        :param frame_height: Height size of the original frames
        :return:
        """
        flow_estimated = []
        frame = None

        for i in range(0, flow.shape[0], frame_height):
            frame = flow[i:(frame_height + i)]
            frame = np.array(frame, dtype=np.uint8)
            frame = cv2.resize(frame, None,
                               fx=1/self.reduce_factor,
                               fy=1/self.reduce_factor,
                               interpolation=cv2.INTER_LINEAR)
            frame = np.expand_dims(frame, axis=0)
            flow_estimated.append(frame)

        flow_estimated.append(frame)
        flow_estimated = np.vstack(flow_estimated).astype(np.float32)

        flow_estimated = self._normalize(flow_estimated)
        return flow_estimated

    def _normalize(self, video):
        # Normalize between -1 and 1
        video /= 127.5
        video -= 1.0
        video = np.expand_dims(video, axis=0)
        return video

    def computeFlow(self, video):
        video_list = self._preprocess(video)
        frame_height = video_list[0].shape[0]

        if self.concatenate:
            # Frame concatenation
            previous, next = self._concatFrames(video_list)

           # start_time_a = time.time()
            # Estimation of the Optical Flow only once
            flow = self.computeTVL1(previous, next)
           # print(time.time()-start_time_a)
            flow_estimated = self._reconstructEstimated(flow, frame_height)
            return flow_estimated

        else:
            # Estimate Optical Flow between every pair of frames
            flow_estimated = []
            flow = []
            for i in range(0, len(video_list) - 1, 1):
                # Estimate flow between a pair of frames
                flow = self.computeTVL1(video_list[i], video_list[i+1])

                # Resize to original size
                flow = flow.astype(np.uint8)
                flow = cv2.resize(flow, None,
                                  fx=1 / self.reduce_factor,
                                  fy=1 / self.reduce_factor,
                                  interpolation=cv2.INTER_LINEAR)
                flow = np.expand_dims(flow, axis=0)
                flow_estimated.append(flow)
            flow_estimated.append(flow)
            flow_estimated = np.vstack(flow_estimated).astype(np.float32)

            # Normalize between -1 and 1
            flow_estimated = self._normalize(flow_estimated)
            return flow_estimated


class FlowEstimatorTVL1OpenCL(FlowEstimatorTVL1):
    def __init__(self, skip_frames=0, reduce_factor=0.5, concatenate=True):
        cv2.ocl.setUseOpenCL(True)

        if not cv2.ocl.useOpenCL():
            raise Exception("OpenCL is not enabled in this device")

        self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(nscales=3,
                                                           innnerIterations=100,
                                                           warps=4)
        super(FlowEstimatorTVL1OpenCL, self).__init__(skip_frames=skip_frames,
                                                      reduce_factor=reduce_factor,
                                                      concatenate=concatenate)

    def computeTVL1(self, previous, next, bound=20):
        # Start converting previous frames to UMat

        prev_frame = cv2.UMat(previous)

        # Start converting next frames to UMat
        next_frame = cv2.UMat(next)

        # Start computing TVL1 OpenCL
        flow = self.tvl1.calc(prev_frame, next_frame, None)

        # Download flow computation from UMat
        flow = cv2.UMat.get(flow)

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)

        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        return flow


class FlowEstimatorTVL1CPU(FlowEstimatorTVL1):
    def __init__(self, skip_frames=0, reduce_factor=0.5, concatenate=True):

        self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create(nscales=3,
                                                           innnerIterations=100,
                                                           warps=4)
        super(FlowEstimatorTVL1CPU, self).__init__(skip_frames=skip_frames,
                                                   reduce_factor=reduce_factor,
                                                   concatenate=concatenate)

    def computeTVL1(self, previous, next, bound=20):
        # Start computing TVL1 OpenCL
        flow = self.tvl1.calc(previous, next, None)

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)

        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        return flow


class FlowEstimatorTVL1CUDA(FlowEstimatorTVL1):
    def __init__(self, skip_frames=0, reduce_factor=0.5, concatenate=True):
        self.tvl1 = cv2.cuda.OpticalFlowDual_TVL1_create(nscales=3,
                                                         iterations=100,
                                                         warps=4)

        super(FlowEstimatorTVL1CUDA, self).__init__(skip_frames=skip_frames,
                                                    reduce_factor=reduce_factor,
                                                    concatenate=concatenate)

    def computeTVL1(self, previous, next, bound=20):
        # Upload frame to GPU to be computed by CUDA
        prev_gpu = cv2.cuda_GpuMat()
        prev_gpu.upload(previous)

        # Upload other frame to the GPU memory
        next_gpu = cv2.cuda_GpuMat()
        next_gpu.upload(next)

        # Estimate flow with CUDA
        flow = self.tvl1.calc(prev_gpu, next_gpu, None)
        flow = flow.download()

        flow = (flow + bound) * (255.0 / (2 * bound))
        flow = np.round(flow).astype(int)

        flow[flow >= 255] = 255
        flow[flow <= 0] = 0

        return flow
