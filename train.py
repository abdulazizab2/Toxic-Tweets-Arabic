import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('toxic arabic tweets classification.txt', '\t') # df columns must be [Tweet, Class]
classes = ['normal', 'abusive', 'hate']


from preprocess import Preprocessor
preprocessor = Preprocessor(df, classes)
preprocessor.map_labels()
df['Tweet'] = df['Tweet'].apply(preprocessor.normalize_text)
maxlen = 25


from model import ToxicModel

toxic_model = ToxicModel(df, maxlen)
train, val, test = toxic_model.load_dataset()
model = toxic_model.build_model()

if __name__ == '__main__':
    opt = tf.keras.optimizers.Adam(1e-4)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy()
    model.compile(optimizer=opt, loss=loss, metrics=[acc])

    history = model.fit(x=train, validation_data=val, epochs=50)
    model.save('models/Bert-Multi-Dialect')
