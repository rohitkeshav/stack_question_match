# use MultinomialNB algorithm
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


def Multinomial(data):

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'],data.p_lang, test_size=0.2)
    

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('clf', MultinomialNB(alpha=1, class_prior=None, fit_prior=True))])

    model = pipeline.fit(X_train, y_train)

    clas_pred = model.predict(X_test)
    # print(clas_pred)

    print("accuracy score - Multinomial: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - MultinomialNB - ','\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - MultinomialNB - ','\n',classification_report(y_test,clas_pred))
    samp_classification_report = classification_report(y_test,clas_pred)
    
    plot_classification_report(samp_classification_report)
    

def Linear_SVC(data, ques):
    
    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'],data.p_lang, test_size=0.1)
    
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('clf', LinearSVC(C=1.0, penalty='l2', max_iter=3000, dual=False,random_state=0))])

    model = pipeline.fit(X_train, y_train)
    
    clas_pred = model.predict(X_test)
    # print(clas_pred)
    
    print("accuracy score - LinearSVC: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - LinearSVC - ','\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - LinearSVC - ','\n',classification_report(y_test,clas_pred))

    predicted = model.predict([ques])

    print('Classified label using LinearSVC - ')
    print('Question - {0}'.format(ques))
    print('Predicted Label - {0}'.format(predicted[0]))
    print('\n')

    samp_classification_report = classification_report(y_test, clas_pred)
    
    plot_classification_report(samp_classification_report)

    return predicted[0]


def Logistic_Regression(data):

    stemmer = SnowballStemmer('english')
    words = stopwords.words("english")

    data['cleaned'] = data['title'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    X_train, X_test, y_train, y_test = train_test_split(data['cleaned'],data.p_lang, test_size=0.1)
    

    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('clf', linear_model.LogisticRegression())])
    

    model = pipeline.fit(X_train, y_train)

    clas_pred = model.predict(X_test)
    # print(clas_pred)

    print("accuracy score - Logistic Reg: " + str(model.score(X_test, y_test)))
    print('Confusion Matrix - Logistic Reg - ','\n',metrics.confusion_matrix(y_test,clas_pred))
    print('Classification Report - Logistic Reg - ','\n',classification_report(y_test,clas_pred))
       
    samp_classification_report = classification_report(y_test,clas_pred)
    
    plot_classification_report(samp_classification_report)
    
 
def plot_classification_report(classification_report, title='Classification report ', cmap='RdBu'):
    lines = classification_report.split('\n')

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[2 : (len(lines) - 2)]:
        t = line.strip().split()
        if len(t) < 2: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)


    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 25
    figure_height = len(class_names) + 7
    correct_orientation = False
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
    

    # Plot it out
    fig, ax = plt.subplots()    
    #c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap='RdBu', vmin=0.0, vmax=1.0)
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    # set tick labels
    #ax.set_xticklabels(np.arange(1,AUC.shape[1]+1), minor=False)
    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    # set title and x/y labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    # Remove last blank column
    plt.xlim( (0, AUC.shape[1]) )

    # Turn off all the ticks
    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    # Add color bar
    plt.colorbar(c)

    # Add text in each cell 
#    show_values(c)

    # Proper orientation (origin at the top left instead of bottom left)
    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    # resize 
    fig = plt.gcf()
    #fig.set_size_inches(cm2inch(40, 20))
    #fig.set_size_inches(cm2inch(40*4, 20*4))
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def cm2inch(*tupl):
    '''
    Specify figure size in centimeter in matplotlib
    Source: https://stackoverflow.com/a/22787457/395857
    By gns-ank
    '''
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)
      
       

if __name__ == "__main__": 
 

    df = pd.read_csv("./data_set.csv",header=None,names=['title','tags','creation_date','username','up_votes','link','p_lang'])
    print('Sample analysis -')
    Multinomial(df)
    Linear_SVC(df)
    Logistic_Regression(df)
