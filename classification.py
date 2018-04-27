# use MultinomialNB algorithm
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report


class ClassifyStackData:
    # title, p_num
    def __init__(self, fname, text, label, cval):
        self.stack_data = pd.read_csv(fname)

        self.text = text
        self.x = self.stack_data[text]
        self.y = self.stack_data[label]
        self.cval = cval

    def fit_data(self):

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, random_state=2)

        c_vect = CountVectorizer(lowercase=True, stop_words='english')

        c_vect.fit(x_train)

        x_train_dtm = c_vect.fit_transform(x_train)
        x_test_dtm = c_vect.transform(x_test)

        return c_vect, x_train_dtm, x_test_dtm, y_train, y_test

    def multinomial_nb(self):

        nb = MultinomialNB()

        return self.predict(nb)

    def logistic_regression(self):

        lr = LogisticRegression()

        return self.predict(lr)

    def linear_svc(self):

        lsv = LinearSVC()

        return self.predict(lsv)

    def check_text(self):

        if type(self.cval) == str:
            return pd.DataFrame({self.text: [self.cval]}, index=[0])

        return pd.DataFrame({self.text: self.cval}, index=[idx for idx in range(len(self.cval))])

    def predict(self, c_obj):
        vect, x_train_dtm, x_test_dtm, y_train, y_test = self.fit_data()

        c_obj.fit(x_train_dtm, y_train)
        y_pred_class = c_obj.predict(x_test_dtm)

        print(c_obj.predict(vect.transform(self.check_text())))
        print(metrics.accuracy_score(y_test, y_pred_class))


def multinomial(data):

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.p_lang, test_size=0.1)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('clf', MultinomialNB(alpha=1, class_prior=None, fit_prior=True))])

    model = pipeline.fit(X_train, y_train)

    clas_pred = model.predict(X_test)

    print(clas_pred)
    print("accuracy score - Multinomial: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - MultinomialNB - ','\n',metrics.confusion_matrix(y_test,clas_pred))
    # print('Classification Report - MultinomialNB - ','\n',classification_report(y_test,clas_pred))
        
    
def linear_svc(data, ques):

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.p_lang, test_size=0.1)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=1500)),
                         ('clf', LinearSVC(C=1.0, penalty='l2', max_iter=3000, dual=False, random_state=0))])

    model = pipeline.fit(X_train, y_train)

    clas_pred = model.predict(X_test)
    print(clas_pred)
    
    print("accuracy score - LinearSVC: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - LinearSVC - ', '\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - LinearSVC - ', '\n',classification_report(y_test,clas_pred))

    return model.predict([ques])[0]


def logistic_regression(data):
    
    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.p_lang, test_size=0.1)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=1500)),
                         ('clf', LogisticRegression())])

    model = pipeline.fit(X_train, y_train)

    clas_pred = model.predict(X_test)
    print(clas_pred)

    print("accuracy score - Logistic regression : " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - Logistic regression - ', '\n', metrics.confusion_matrix(y_test,clas_pred))
    # print('Classification Report - Decision Tree - ','\n',classification_report(y_test,clas_pred))


# logistic regression
"""
def __logistic_regression(df):
    stack_data = pd.read_csv('./data_set.csv')

    # define X, y

    X = stack_data.title
    y = stack_data.p_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    vect = CountVectorizer(lowercase=True, stop_words='english')

    vect.fit(X_train)

    # transform training data
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    nb = LogisticRegression()

    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print(nb.predict(vect.transform(test_case())))

    print(metrics.accuracy_score(y_test, y_pred_class))
"""


def test_case():
    return pd.DataFrame({'title': ["Does Python have a string 'contains' substring method?"]}, index=[i for i in range(len(["Does Python have a string 'contains' substring method?"]))])


if __name__ == "__main__":
    df = pd.read_csv("data_set.csv")
    print('Classifiers -')

    print(logistic_regression(df))
    # csd = ClassifyStackData('./data_set.csv', 'title', 'p_num', [''])

    # print(csd.multinomial_nb())
    # print(csd.logistic_regression())
    # print(csd.linear_svc())
