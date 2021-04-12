# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PBV1ow7d_dWqkcz3HBtWqyFaQWHJTBRW
"""

!pip install tensorflow==2.0.0

!pip install tensorflow-gpu==2.0.0

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
tf.__version__

!git clone https://github.com/carolinedunn/face_mask_detection

!cd /content/face_mask_detection && python3 train_smoking_detector.py --dataset dataset --plot mymodelplot.png --model my.model

from google.colab import files
files.download('content/face_mask_detection/my.model')