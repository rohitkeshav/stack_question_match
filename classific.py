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
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def multinomial(data):

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.p_lang, test_size=0.1)

    print(TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).get_feature_names())
    import time
    time.sleep(500)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=200)),
                         ('clf', MultinomialNB(alpha=1, class_prior=None, fit_prior=True))])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vect']
    chi = model.named_steps['chi']
    clf = model.named_steps['clf']
    
    clas_pred = model.predict(X_test)
    print(clas_pred)
    print("accuracy score - Multinomial: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - MultinomialNB - ','\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - MultinomialNB - ','\n',classification_report(y_test,clas_pred))
        
    
def linear_svc(data):

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.p_lang, test_size=0.1)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=1500)),
                         ('clf', LinearSVC(C=1.0, penalty='l2', max_iter=3000, dual=False, random_state=0))])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vect']
    chi = model.named_steps['chi']
    clf = model.named_steps['clf']
    
    clas_pred = model.predict(X_test)
    print(clas_pred)
    
    print("accuracy score - LinearSVC: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - LinearSVC - ', '\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - LinearSVC - ', '\n',classification_report(y_test,clas_pred))


def decision_tree(data):
    
    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower()) #his
    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i.lower() not in words]).lower()) #mine

    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'], data.p_lang, test_size=0.1)

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi',  SelectKBest(chi2, k=1500)),
                         ('clf', DecisionTreeClassifier(random_state=0))])

    model = pipeline.fit(X_train, y_train)

    vectorizer = model.named_steps['vect']
    chi = model.named_steps['chi']
    clf = model.named_steps['clf']

    clas_pred = model.predict(X_test)
    print(clas_pred)

    print("accuracy score - DecisionTree: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - Decision Tree - ','\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - Decision Tree - ','\n',classification_report(y_test,clas_pred))


def __try():
    features = ['p_lang', 'title', 'p_num']
    stack_data = pd.read_csv('./data_set.csv')

    # define X, y
    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")
    X = stack_data.title.apply(
        lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    #X = stack_data.title
    y = stack_data.p_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    vect = CountVectorizer(lowercase=True, stop_words='english')

    vect.fit(X_train)

    # transform training data
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    nb = MultinomialNB()

    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    # print(nb.predict(vect.transform(testcase())))
    print(metrics.accuracy_score(y_test, y_pred_class))


# Linear SVC
def _c_try(testdata = None):
    features = ['p_lang', 'title', 'p_num']
    stack_data = pd.read_csv('./data_set.csv')

    # define X, y
    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")
    X = stack_data.title.apply(
        lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    #X = stack_data.title
    y = stack_data.p_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    vect = CountVectorizer(lowercase=True, stop_words='english')

    vect.fit(X_train)

    # transform training data
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    lsv = LinearSVC()

    lsv.fit(X_train_dtm, y_train)
    y_pred_class = lsv.predict(X_test_dtm)
    print('hey')
    print(stack_data.p_lang.unique())
    print(metrics.accuracy_score(y_test, y_pred_class))

    if testdata:
        testdata.title = testdata.title.apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
        print(lsv.predict(vect.fit_transform(testdata.title)))

    #print(X_test.shape, X_test_dtm.shape, X_train_dtm.shape, X_train.shape)


# Random forest
def _d_try():
    features = ['p_lang', 'title', 'p_num']
    stack_data = pd.read_csv('./data_set.csv')


    # define X, y
    X = stack_data.title
    y = stack_data.p_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    vect = CountVectorizer(lowercase=True, stop_words='english')

    vect.fit(X_train)

    # transform training data
    X_train_dtm = vect.fit_transform(X_train)
    y_train_dtm = vect.fit_transform(y_train)
    X_test_dtm = vect.transform(X_test)

    rf = RandomForestClassifier(n_estimators= 1000, random_state= 40)
    rf.fit(X_train_dtm, y_train_dtm)
    prediction = rf.predict(X_test)
    print(prediction)

    #scores = cross_validate(rf, X_train_dtm, y_train_dtm, cv=100, return_train_score=True)

    #print(scores)


# logistic regression
def _l_try():
    features = ['p_lang', 'title', 'p_num']
    stack_data = pd.read_csv('./data_set.csv')

    # define X, y

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")
    X = stack_data.title.apply(
        lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    #X = stack_data.title
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
    # print(nb.predict(vect.transform(testcase())))

    print(metrics.accuracy_score(y_test, y_pred_class))


# Neural Nets
def __nn_try(hidden_layer_size):
    features = ['p_lang', 'title', 'p_num']
    stack_data = pd.read_csv('./data_set.csv')

    # define X, y
    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")
    X = stack_data.title.apply(
        lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    #X = stack_data.title
    y = stack_data.p_num

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    vect = CountVectorizer(lowercase=True, stop_words='english')

    vect.fit(X_train)

    # transform training data
    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = hidden_layer_size, random_state = 1)



    clf.fit(X_train_dtm, y_train)
    y_pred_class = clf.predict(X_test_dtm)
    # print(nb.predict(vect.transform(testcase())))

    print(hidden_layer_size, metrics.accuracy_score(y_test, y_pred_class))


def _c_try_mod(testdata=None):

    print('\n\n Modded SVM code')

    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer(r'\w+')
    stack_data = pd.read_csv('./data_set.csv')


    # Define X and y
    X = stack_data.title.apply(lambda x: ' '.join(
        [stemmer.stem(i.lower()) for i in tokenizer.tokenize(x) if i.lower() not in stopwords.words("english")]))
    y = stack_data.p_lang

    # Test and Train Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    vect = CountVectorizer(lowercase=True)
    vect.fit(X_train)

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    lsv = LinearSVC()
    lsv.fit(X_train_dtm, y_train)
    y_pred_class = lsv.predict(X_test_dtm)
    print(metrics.accuracy_score(y_test, y_pred_class))

    if testdata is not None:
        testdata.title = testdata.title.apply(lambda x: ' '.join(
            [stemmer.stem(i.lower()) for i in tokenizer.tokenize(x) if i.lower() not in stopwords.words("english")]))
        #print(lsv.predict(vect.transform(testdata.title)))
        #print(testdata.title.values)
        for i, j in zip(testdata.title.values, lsv.predict(vect.transform(testdata.title))):
            print(i, ' | predicted as : ', j)


def testcase(stringList):
    return pd.DataFrame({'title': stringList}, index=[i for i in range(len(stringList))])


if __name__ == "__main__":

    df = pd.read_csv("data_set.csv")
    print('Sample analysis -')
    # multinomial(df)
    _c_try()

    s = ['What is abstract class in Java']
    _c_try_mod(testcase(s))
    # _d_try()
    _l_try()
    __try()
    __nn_try((20)) # Pass number and size of hidden layers as tuple inputs
    __nn_try((40))
    __nn_try((20, 10))
    __nn_try((10, 5))
    __nn_try((20, 10, 5))
    # linear_svc(df)
    # decision_tree(df)
