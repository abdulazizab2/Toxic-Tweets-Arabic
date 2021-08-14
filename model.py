import numpy as np
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModel




class ToxicModel:

    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
    bert = TFAutoModel.from_pretrained("asafaya/bert-base-arabic")
    maxlen = 25
    
    def __init__(self, df, maxlen):
        self.df = df
        self.maxlen = maxlen
        self.arr = self.df['Class'].values
        self.labels = np.zeros((self.arr.size, self.arr.max()+1))
        self.labels[np.arange(self.arr.size), self.arr] = 1
        self.Xids = np.zeros((len(self.df), self.maxlen))
        self.Xmask = np.zeros((len(self.df), self.maxlen))

    def _tokenize(self, sentence):
        tokens = ToxicModel.tokenizer.encode_plus(sentence, max_length=self.maxlen,
                                    truncation=True, padding='max_length',
                                    add_special_tokens=True, return_attention_mask=True,
                                    return_token_type_ids=False, return_tensors='tf')
        return tokens['input_ids'], tokens['attention_mask']

    def tokenize(self):
        for i, sentence in enumerate(self.df['Tweet']):
            self.Xids[i, :], self.Xmask[i, :] = self._tokenize(sentence)

    @staticmethod
    def map_func(input_ids, masks, labels):
        return {'input_ids': input_ids, 'attention_masks': masks}, labels

    def load_dataset(self, batch_size=64):
        self.tokenize()
        self.dataset = tf.data.Dataset.from_tensor_slices((self.Xids, self.Xmask, self.labels))
        self.dataset = self.dataset.map(ToxicModel.map_func)
        self.dataset = self.dataset.shuffle(self.Xids.shape[0]*2).batch(batch_size)
        DS_LEN = len(list(self.dataset))
        train = self.dataset.take(round(DS_LEN*0.8))
        temp = self.dataset.skip(round(DS_LEN*0.8))
        val = temp.take(round(len(list(temp))*0.5))
        test = temp.skip(round(len(list(temp))*0.5))
        del temp
        return train, val, test

    @classmethod
    def build_model(cls):
        input_ids = tf.keras.layers.Input(shape=(cls.maxlen, ), name='input_ids', dtype='int32')
        mask = tf.keras.layers.Input(shape=(cls.maxlen, ), name='attention_masks', dtype='int32')

        embeddings = cls.bert(input_ids, mask)[0]


        X = tf.keras.layers.BatchNormalization()(embeddings)
        X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, recurrent_dropout=0.2, return_sequences=False, name='LSTM'))(X)
        X = tf.keras.layers.Dense(128, 'relu', name='FC1')(X)
        X = tf.keras.layers.Dropout(0.3)(X)
        X = tf.keras.layers.Dense(64, 'relu', name='FC2')(X)
        X = tf.keras.layers.BatchNormalization()(X)
        output = tf.keras.layers.Dense(units=3, activation='softmax', name='output')(X)

        model = tf.keras.Model(inputs=[input_ids, mask], outputs=output)
        model.layers[2].trainable = False

        return model
