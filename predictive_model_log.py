import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Combined_News_DJIA.csv')

def remove_b(a):
    if ((a is not np.nan) and (a[0] == 'b')):
        return a[1:]
    else:
        return a

for c in data.columns[2:]:
    data[c] = data[c].apply(remove_b)

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

# Cleaning the texts
print('----------------Cleaning the texts------------------')
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean(corpus):
    c = []
    for i in range(len(corpus)):
        text = re.sub('[\.$]', '', corpus[i])
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = ' '.join(text)
        c.append(text)
    return c

train_corpus = clean(trainheadlines)
test_corpus = clean(testheadlines)

def remove_stopwords(corpus):
    for i in range(len(corpus)):
        c = corpus[i].split(' ')
        #c = [word for word in c if word not in stop_words]
        corpus[i] = c
    return corpus
train_corpus = remove_stopwords(train_corpus)
test_corpus = remove_stopwords(test_corpus)

# Stemming the words
from nltk.stem import PorterStemmer, SnowballStemmer
#stemmer=SnowballStemmer(language='english',ignore_stopwords=False)
stemmer=PorterStemmer()
train_corpus=[' '.join([stemmer.stem(word) for word in text])
          for text in train_corpus]

test_corpus=[' '.join([stemmer.stem(word) for word in text])
          for text in test_corpus]

# Creating an ngram model
from sklearn.feature_extraction.text import CountVectorizer
num_features = 235
vectorizer = CountVectorizer(ngram_range=(2,2),max_features=num_features)
X_train = vectorizer.fit_transform(train_corpus)
y_train = train['Label'].astype(int)
X_test = vectorizer.transform(test_corpus)
y_test = test['Label'].astype(int)

# Making Predictions
print('----------------Making Predictions-----------------')
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(solver='sag',max_iter=500, C=0.31)
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

# Evaluating model
from sklearn.metrics import roc_curve,roc_auc_score,classification_report,confusion_matrix
import scikitplot as skplt
cm = confusion_matrix(y_test, y_pred)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print('Accuracy: ', accuracy)
skplt.metrics.plot_confusion_matrix(y_test, y_pred)

print('\n',classification_report(y_test, y_pred))

y_pred_proba = log_reg.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

# Rectification analysis
def show_most_informative_features(vectorizer, clf, n):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print ('\t%.4f\t%-15s\t\t%.4f\t%-15s' % (coef_1, fn_1, coef_2, fn_2))

print('\n')
show_most_informative_features(vectorizer, log_reg, 15)
print('\n')