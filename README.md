# Efficient Reconfigurable System for Home Monitoring of the Elderly via Action Recognition

This repository of from the work: Efficient Reconfigurable System for Home Monitoring of the Elderly via Action Recognition [1]()

Here, we provide the Deep Learning models architectures using the [Keras]() framework and the weights of the trained models.

## Pre-requisites
Firstly, create a python virtualenv to run these models. Then activate the python virtual environment. It is required
a Python version `>=3.10` to run this repo.

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

## Demo tutorial
Refer to [demo_tutorial.ipynb](demo_tutorial.ipynb) to see an example of how to load the models introduced with their
weights and how to do inferences to perform action recognition.

This tutorial shows the recognition confidence of the solution when analyzing a critical action using the most efficient,
and the most computational intensive trained model architectures.

## Citation
[1] D. Deniz, J. Isern, J. Solanti, P. Jääskeläinen, P. Hnětynka, L. Bulej, E. Ros, and F. Barranco. 
"Efficient Reconfigurable System for Home Monitoring of the Elderly via Action Recognition" in Engineering Applications of Artificial Intelligence.

## License
[BSD 3-Clause License](LICENSE)