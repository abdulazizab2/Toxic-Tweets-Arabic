import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from utils import tensorflow_to_numpy
from train import *

model.load_weights('models\Bert-Multi-Dialect')
true_classes, predicted_classes = tensorflow_to_numpy(ds=test, model=model)