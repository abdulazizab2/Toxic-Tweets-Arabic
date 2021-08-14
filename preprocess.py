import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pyarabic.araby as araby
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_list = stopwords.words('arabic')

class Preprocessor():

    def __init__(self, df, classes):
        self.df = df
        self.classes = classes

    def ClassCount(self, figsize=(10,6)):
        frequency = []
        for label in self.classes:
            frequency.append(len(self.df[self.df.Class==label]))
        self.counter = dict(zip(self.classes, frequency))
        plt.figure(figsize=figsize)
        sns.barplot(list(self.counter.keys()), list(self.counter.values()), alpha=1)
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.title('Number of Comments Per Class')
        plt.show()
  
        return self.counter

    def map_labels(self):
        # change labels if classes are different than 3
        self.df['Class'] = self.df['Class'].replace({'normal':0, 'abusive':1, 'hate':2})

    @staticmethod
    def normalize_text(text):

        text = re.sub('[^\u0600-\u06FF\s!؟]', '', text) # remove non-arabic ASCII and keep !?
        text = re.sub('[٠-٩]', '', text) # remove arabic numbers
        text = re.sub('[.،:-؛()]', '', text) # remove punctuation marks except ? and !
        text = re.sub('[اآأإٱىٰ]', 'ا', text)  # subsitutes forms of alif with one form
        text = araby.strip_diacritics(text) # remove diacritics
        text = re.sub(' +', ' ', text) # ensure only one empty space between words
        text = ' '.join([tok for tok in text.split() if tok not in (stopwords_list)])
        text = text.strip() # remove leading and trailing spaces
        return text


    def density_plot(self):
        seqlen = self.df['Tweet'].apply(lambda x: len(x.split(' ')))
        plt.figure(figsize=(10,8))
        sns.distplot(seqlen)
        plt.show()