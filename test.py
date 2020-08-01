import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error
import math

data = pd.read_csv('shuffled-full-set-hashed.csv', names=['cat
egory','words'])

data1 = data.dropna(subset=['words'])
data1 = data1.reset_index()
data1 = data1.drop(['index'],axis=1)
#Splitting the data into training and validation set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data1['words'],data1['category'],test_size=0.3,random_state=42)
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

text_clf_svm = text_clf_svm.fit(x_train, y_train)
predicted_svm = text_clf_svm.predict(x_test)
#np.mean(predicted_svm == y_test)
import pickle
pickle.dump(text_clf_svm, open('SGDClassiferi.pkl', 'wb'))