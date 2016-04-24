import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# download data here: http://www.cpsc.gov/en/Research--Statistics/NEISS-Injury-Data/
file_path = '/Users/joshuamalina/Downloads/NEISS-data-2014-updated-12MAY2015.csv'

'''
        # genderedWords = ["him", "her", "his", "he", "she", "male", "female",
        # "himself", "herself", "man", "woman", "penis", "scrotum", "vagina", "clitoris"]
'''


class Injury(object):
    def __init__(self, datapath):
        columns_to_keep = ['sex', 'age', 'narrative']
        self.datapath = datapath
        print("loading data from file ...")
        self.raw = pd.read_csv(datapath)
        self.data = self.raw[columns_to_keep]
        print("encoding gender ...")
        self.data['binary_sex'] = self.make_gender_binary(self.data.sex)
        print("transforming age ...")
        self.data['age_transformed'] = self.transform_ages(self.data.age)  # returns some null when age is unknown
        self.data = self.data.dropna()
        print("building tfidf matrix ...")
        self.vectorizer = TfidfVectorizer(norm='l1', stop_words=[])
        self.tfidfmatrix = self.vectorizer.fit_transform(self.data.narrative)
        print("initiation complete")

    # encode gender as 1 or 0
    @staticmethod
    def make_gender_binary(sex):
        gender_legend = {"Male": 0, "Female": 1}
        return pd.Series([gender_legend["Male"] if x == "Male" else gender_legend["Female"] for x in sex])

    # ages are in years, but for children < 1 years, it is funky
    def transform_ages(self, ages):
        return pd.Series([self.transform_age(x) for x in ages])

    # converts a single age to an age that makes sense, if age is unknown (i.e. age == 0) return None
    @staticmethod
    def transform_age(age):
        if age == 0:
            return None
        elif age < 200:
            return age
        else:
            stripped = int(str(age)[1:])  # grabs number of months
            return stripped / 12.0
