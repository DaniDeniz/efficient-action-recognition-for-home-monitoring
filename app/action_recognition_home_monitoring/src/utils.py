import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_video(video_path: str, height: int = 112, width: int = 112,
               frames_wanted: int = 64, step: int = 64,
               shortside: int = 112):
    """
    Load a video from a file and preprocess it to analyze it with RGBI3D model
    :param video_path: path of the video
    :param height: desired video height
    :param width: desired video width
    :param frames_wanted: number of frames desired
    :param step: moving window step size
    :param shortside: shortside of the video desired before cropping h x w
    :return: Preprocessed video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Check H and W and get proportion to establish the shortside
    video_h, video_w, resize, por = __video_size(video_h, video_w, shortside)

    init_h, init_w, final_h, final_w = __get_window_coord(video_h, video_w, height, width, True)
    video1 = []
    current_frame = 0
    final_frame = n_frames

    while current_frame + frames_wanted <= final_frame:
        frames_recorded = 0
        window = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        while frames_recorded < frames_wanted:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=por, fy=por)
                frame_rgb = cv2.cvtColor(frame[init_h:final_h, init_w:final_w],
                                         cv2.COLOR_BGR2RGB)
                window.append(frame_rgb)
                frames_recorded = frames_recorded + 1
            else:
                break
        video1.append(window)
        current_frame = current_frame + step

    del window
    cap.release()
    del cap

    video = np.array(video1, dtype=np.float32)
    del video1

    return video


def scale_video(x: np.ndarray, type="-1_1"):
    """
    Scale video of numpy array
    :param x: numpy array
    :param type: normalization type (between -1 and 1 / 0 and 1)
    :return: normalized video
    """
    x = x.astype(np.float32)
    if type == "-1_1":
        x /= 127.5
        x -= 1.
    else:
        x /= 255.
    return x


def __video_size(video_h, video_w, shortside):
    """
    Check the video size and return the new values to get the wanted
    video shortside
    :param video_h: Original video height
    :param video_w: Original video width
    :param shortside: shortside
    :return: New video height and width, and also a boolean telling if
    resize is necessary and proportion
    """
    por = 1.0
    resize = False
    if video_h != shortside or video_w != shortside:
        resize = True
        if video_h < video_w:
            por = float(shortside) / video_h
        else:
            por = float(shortside) / video_w
        video_h = int(video_h*por)
        video_w = int(video_w*por)

    return video_h, video_w, resize, por


def __get_window_coord(video_h: int, video_w: int, window_h: int, window_w: int, test: bool):
    """
    Obtain a window coordinates to crop the video
    :param video_h: Current video height
    :param video_w: Current video width
    :param window_h: Desired window height size
    :param window_w: Desired window width size
    :param test: Boolean, crop a centered (True) or random (False) window
    :return: Window crop positions: height and width position
    """
    if test:
        init_h = int(video_h / 2 - window_h / 2)
        init_w = int(video_w / 2 - window_w / 2)

    else:
        init_h = np.random.randint(0, video_h - window_h + 1)
        init_w = np.random.randint(0, video_w - window_w + 1)

    final_h = init_h + window_h
    final_w = init_w + window_w

    return init_h, init_w, final_h, final_w


def show_video(video, frames=5):
    i = 0
    fig, axs = plt.subplots(1, frames, figsize=(20, 20))

    for k in np.linspace(0, 64, frames).astype(np.uint8):
        if k >= video.shape[0]:
            k = video.shape[0] - 1
        axs[i].imshow(np.uint8(video[k]))
        axs[i].axis('off')
        i += 1
