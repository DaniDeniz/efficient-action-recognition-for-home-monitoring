# Efficient Reconfigurable System for Home Monitoring of the Elderly via Action Recognition

This repository is from the work: Efficient Reconfigurable System for Home Monitoring of the Elderly via Action Recognition [[1]()]

We provide **Deep Learning model architectures and pre-trained weights** using [Keras](https://keras.io/api/) 
for action recognition in elderly home monitoring scenarios.


## 📦 Overview  
Our system leverages edge computing to enable real-time, non-invasive monitoring of elderly individuals through 
activity recognition. Key features include:  
- **Lightweight AI models** optimized for embedded devices (e.g., Jetson Nano).  
- **Run-time reconfiguration** to adapt to changing resource constraints or environmental conditions.  
- **Pre-trained weights** for both efficient and high-accuracy architectures.  


## 🧰 Prerequisites  
To use this repository:  
1. **Python version**: `>=3.10`.  
2. **Dependencies**: Install via virtual environment (recommended).  


```bash
python3 -m virtualenv -p python3 venv
source venv/bin/activate
```

Clone this repository to your machine
```bash
git clone https://github.com/DaniDeniz/efficient-action-recognition-for-home-monitoring.git && cd efficient-action-recognition-for-home-monitoring
```

Install the `action_recognition_home_monitoring` Python package

```bash
pip install -e .
```

> 📌 Note: Ensure your environment has CUDA and cuDNN installed if using GPU acceleration.  
> 🔍 Requirements for the demo notebook: `jupyter`, `matplotlib`, `opencv-python`, `numpy`, `tensorflow`, and `keras`.


## 🚀 Getting Started  

### 1. **Model Usage**  
Pre-trained models are available in the `models_weights/` directory. Load them directly via the `action_recognition_home_monitoring` package:

```python
from action_recognition_home_monitoring import RGBI3D, TwoStream

# Example: Load a pre-trained model (replace with your chosen architecture)
video_model = RGBI3D(model_keras_file="./models_weights/RGBI3D_16_112.keras")
two_stream_model = TwoStream(
    rgb_model_keras_file="./models_weights/RGBI3D_64_112.keras",
    flow_model_keras_file="./models_weights/OpticalFlow_64_112.keras"
)
```

### 2. **Demo tutorial**
Refer to [demo_tutorial.ipynb](demo_tutorial.ipynb) to see an example of how to load the models introduced with their
weights and how to do inferences to perform action recognition.

This tutorial shows the recognition confidence of the solution when analyzing a critical action using the most efficient,
and the most computational intensive trained model architectures.

#### 🎥 Sample Workflow in `demo_tutorial.ipynb`  

1. **Load Pretrained Model Weights**:  
   - Example: `RGBI3D_16_112.keras`, `RGBI3D_32_112.keras`, `RGBI3D_64_112.keras`, or `OpticalFlow_64_112.keras`.

2. **Load and Preprocess Input Video**:  
   ```python
   from action_recognition_home_monitoring import load_video
   video_file_name = "action_fall_sample.mp4"
   video_path = "./res/video/{}".format(video_file_name)
   video_clips = load_video(video_path)  # Returns clips of 64 frames (2.56 sec at 25 FPS)
   ```

3. **Show the Loaded Video**:  
   ```python
   from action_recognition_home_monitoring import show_video
   show_video(video_clips[0], frames=5)  # Visualize first clip with 5 frames
   ```

4. **Infer Activity Using RGBI3D Model (Efficient)**:  
   - Example output:
     ```
     Action recognizer detected: falling down - with a confidence of: 47.21%
     ```

   ```python
   from action_recognition_home_monitoring import load_video, show_video
   import numpy as np
   def predict_data(model, video_path, actions_list):
       prediction = model.predict(video_path, batch_size=8)
       prediction = prediction.mean(axis=0)
       detected_action = np.argmax(prediction, axis=-1)

       print("Action recognized: {} - with a confidence of: {:0.2f}%".format(
           actions_list[detected_action], prediction[detected_action] * 100
       ))

   predict_data(model=video_model, video_path=video_path, actions_list=actions_list)
   ```

5. **Infer Activity Using TwoStream Model (High Accuracy)**:  
   - Example output:
     ```
     Action recognized: falling down - with a confidence of: 98.76%
     ```

   ```python
   predict_data(model=two_stream_model, video_path=video_path, actions_list=actions_list)
   ```

## 📚 Model Details  
- **Available Architectures**:  
  - `RGBI3D_16_112.keras`: Lightweight model for low-power edge devices.  
  - `RGBI3D_32_112.keras`: Balanced model offering a trade-off between efficiency and accuracy. 
  - `RGBI3D_64_112.keras`: High-accuracy model with increased temporal resolution (higher computational cost).  
  - `OpticalFlow_64_112.keras`: Optical flow branch used in the TwoStream architecture for improved recognition confidence.  

- **Sample Data**:  
  - Input video: [./res/video/action_fall_sample.mp4](res/video/action_fall_sample.mp4) (used in the demo).  
  - Action labels: Defined in [./res/classes.json](res/classes.json).

## 📌 Citation
[1] D. Deniz, J. Isern, J. Solanti, P. Jääskeläinen, P. Hnětynka, L. Bulej, E. Ros, and F. Barranco. 
"Efficient Reconfigurable System for Home Monitoring of the Elderly via Action Recognition" in Engineering Applications of Artificial Intelligence.

## 📄 License
[BSD 3-Clause License](LICENSE)