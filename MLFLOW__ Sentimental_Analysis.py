#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis of Real-time Flipkart Product Reviews
# 
# ## Objective:
# 
# ### The objective of this project is to classify customer reviews as positive or negative and understand the pain points of customers who write negative reviews. By analyzing the sentiment of reviews, we aim to gain insights into product features that contribute to customer satisfaction or dissatisfaction.
# 

# # Importing required libraries.

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Reading csv using pandas

# In[35]:


df = pd.read_csv('data.csv')


# # Displaying First 5 rows:

# In[36]:


df.head()


# In[37]:


df.describe()


# In[38]:


df.info()


# # Checking Null values in the data

# In[39]:


df.isnull().sum()


# # Handling Null values:

# In[40]:


df['Review text'].fillna(df['Review text'].mode()[0],inplace=True)


# In[41]:


df.isnull().sum()


# # Check for duplicate values and removing them:

# In[42]:


df.duplicated().sum()


# In[43]:


df = df.drop_duplicates()


# In[44]:


df.head()


# In[45]:


# df.to_csv('data_1.csv')


# # Ploting the Ratings Data:

# In[46]:


sns.countplot(x='Ratings', data=df)


# # Ratings categorised into positive and negative

# In[47]:


df['label'] = df['Ratings'].apply(lambda x: 'positive' if x >= 3 else 'negative')


# # Identifying Input and Output variables

# ### Independent variable

# In[48]:


X=df[['Review text']]


# ### Target/Dependent variable

# In[49]:


y=df[['label']]


# # Train-Test Split

# In[50]:


from sklearn.model_selection import train_test_split


# ## Training data is 80% of total data and Test data is 20% of total data

# In[51]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.8,random_state=35)


# In[52]:


X_train


# In[53]:


y_train


# In[54]:


X_test


# In[55]:


y_test


# # Data Preprocessing of Train data and Test data

# In[56]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[57]:


stem = PorterStemmer()


# In[58]:


lemma = WordNetLemmatizer()


# In[59]:


def preprocess(data):
    
    # removes special characters
    sentence = re.sub("[^a-zA-Z]"," ",data)
    
    # converts words to lowercase
    sentence = sentence.lower()
    
    # tokenization
    sentence = sentence.split()
    
    #removes the stop words
    sentence = [word for word in sentence if word not in stopwords.words('english')]
    
    # can apply stem or lemm
    # applying lemmatization
    sentence = [lemma.lemmatize(word) for word in sentence]
    
    sentence=  " ".join(sentence)
    
    return sentence


# ## Applying preprocesing on train_data

# In[60]:


X_train= X_train['Review text'].apply(preprocess)


# ## Applying preprocesing on test data

# In[61]:


X_test = X_test['Review text'].apply(preprocess)


# In[69]:


df.head()


# # CountVectorizer

# ## Converting Text data to Numerical data

# In[62]:


from sklearn.feature_extraction.text import CountVectorizer


# In[63]:


cv = CountVectorizer()


# In[64]:


get_ipython().run_line_magic('time', 'X_train_num = cv.fit_transform(X_train)')


# In[65]:


get_ipython().run_line_magic('time', 'X_test_num = cv.transform(X_test)')


# # Model Building:

# ## 1. LogisticRegression

# In[32]:


from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(X_train_num, y_train)


# In[33]:


y_train_pred = lr_model.predict(X_train_num)


# In[34]:


y_pred=lr_model.predict(X_test_num)


# In[35]:


from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


# In[36]:


train_score=accuracy_score(y_train,y_train_pred)
train_score


# In[37]:


test_score=accuracy_score(y_pred,y_test)
test_score


# ## 2. Random Forest

# In[38]:


from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train_num, y_train)


# In[39]:


y_train_pred = rf_model.predict(X_train_num)


# In[40]:


y_pred=lr_model.predict(X_test_num)


# In[41]:


from sklearn.metrics import accuracy_score, classification_report

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


# In[42]:


train_score=accuracy_score(y_train,y_train_pred)
train_score


# In[43]:


test_score=accuracy_score(y_pred,y_test)
test_score


# ## 3. Naive Bayes

# In[44]:


from sklearn.naive_bayes import MultinomialNB


# In[45]:


nb_model= MultinomialNB()


# In[46]:


nb_model.fit(X_train_num,y_train)


# ## Evaluation of Naive Bayes Model:

# In[47]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[48]:


y_train_pred = nb_model.predict(X_train_num)
y_train_pred


# In[49]:


train_score=accuracy_score(y_train,y_train_pred)
train_score


# In[50]:


y_pred=nb_model.predict(X_test_num)
y_pred


# In[51]:


test_score=accuracy_score(y_pred,y_test)
test_score


# ## Confusion Matrix

# In[52]:


confusion_matrix(y_train, y_train_pred)


# In[53]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_train, y_train_pred)
actual = np.sum(cm, axis=1).reshape(-1, 1)
cmn = np.round(cm/actual, 2)

sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[54]:


cm = confusion_matrix(y_pred,y_test)
actual = np.sum(cm, axis=1).reshape(-1, 1)
cmn = np.round(cm/actual, 2)

sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# # Implementing various Algorithms to find the Best Model

# In[3]:


from joblib import Memory
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
from sklearn.metrics import f1_score


# In[4]:


import warnings
warnings.filterwarnings('ignore')


# # Defining a memory object to cache Intermediate results

# In[57]:


cachedir = '.cache'
memory = Memory(location=cachedir, verbose=0)

pipelines = {
    'naive_bayes': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', MultinomialNB())
    ], memory=memory),
    'decision_tree': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', DecisionTreeClassifier())
    ], memory=memory),
    'logistic_regression': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', LogisticRegression())
    ], memory=memory)
}

# Define parameter grid for each algorithm
param_grids = {
    'naive_bayes': [
        {
            'vectorization': [CountVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000], 
            'classifier__alpha' : [1, 10]
        }
    ],
    'decision_tree': [
        {
        'vectorization': [CountVectorizer(),TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'logistic_regression': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000], 
            'classifier__C': [0.1, 1, 10], 
            'classifier__penalty': ['elasticnet'], 
            'classifier__l1_ratio': [0.4, 0.5, 0.6],
            'classifier__solver': ['saga'],
            'classifier__class_weight': ['balanced']
        }
    ]
}

# Perform GridSearchCV for each algorithm

best_models = {}

for algo in pipelines.keys():
    print("*" * 10, algo, "*" * 10)
    grid_search = GridSearchCV(estimator=pipelines[algo], 
                               param_grid=param_grids[algo], 
                               cv=5, 
                               scoring='f1',  
                               return_train_score=True,
                               verbose=1)
    
    grid_search.fit(X_train, y_train)
    best_models[algo] = grid_search.best_estimator_
    y_pred = grid_search.best_estimator_.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label='positive')  
    print('F1 Score on Test Data:', f1)


# In[58]:


for name, model in best_models.items():
    print(f"{name}")
    print(f"{model}")
    print()


# In[59]:


import joblib
import os


# In[60]:


from sklearn.metrics import f1_score


# In[61]:


save_dir = 'Models_1'

for name, model in best_models.items():
    print("*" * 10, name, "*" * 10)
    
    joblib.dump(model, os.path.join(save_dir, f'{name}.pkl'))
    loaded_model = joblib.load(os.path.join(save_dir, f'{name}.pkl'))
    
    get_ipython().run_line_magic('time', 'y_test_pred = loaded_model.predict(X_test)')

    f1 = f1_score(y_test, y_test_pred, pos_label='positive')

    print("F1 Score (Positive Class):", f1)
    print("Model Size:", os.path.getsize(os.path.join(save_dir, f'{name}.pkl')), "Bytes")


# In[1]:


# !pip scikit-learn --version


# In[3]:


# pip list


# ##### scikit-learn              1.3.2

# In[4]:


# blinker==1.7.0
# click==8.1.7
# colorama==0.4.6
# flask==3.0.2
# importlib-metadata==7.1.0
# itsdangerous==2.1.2
# Jinja2==3.1.3
# joblib==1.3.2
# MarkupSafe==2.1.5
# numpy==1.26.4
# scikit-learn==1.4.1.post1
# scipy==1.12.0
# threadpoolctl==3.4.0
# werkzeug==3.0.1
# zipp==3.18.1


# In[5]:


# !pip install scikit-learn==1.4.1.post1


# In[6]:


# !pip install joblib==1.3.2


# In[7]:


# !pip install scipy==1.12.0


# In[8]:


# pip list


# In[81]:


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline


# # Experiment Tracking with MLFLOW

# In[82]:


import mlflow

mlflow.set_experiment("sentimental_anaysis")


# # Auto Logging KNN Experiment Run using MLFlow

# In[83]:


# Define pipeline steps
pipe_1 = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]
)


# Observe the Key Value Pair format
parameter_grid_1 = [
    {
        'scaler': [StandardScaler(), MaxAbsScaler()],
        'classifier__n_neighbors' : [i for i in range(3, 21)],              
        'classifier__p' : [1, 2, 3]
    }
]


# In[84]:


import warnings

warnings.filterwarnings('ignore')


# In[85]:


# df = pd.read_csv('data_1.csv')


# In[86]:


# X=df[['Review text']]


# In[87]:


# y=df[['label']]


# In[88]:


clf = GridSearchCV(
    estimator=pipe_1, 
    param_grid=parameter_grid_1, 
    scoring='accuracy',
    cv=5,
    return_train_score=True,
    verbose=1
)

# Initialize the auto logger
# max_tuning_runs=None will make sure that all the runs are recorded.
# By default top 5 runs will be recorded for each experiment
mlflow.sklearn.autolog(max_tuning_runs=None)

with mlflow.start_run() as run:
    get_ipython().run_line_magic('time', 'clf.fit(X_train_num, y_train)')


# In[67]:


# # Define pipeline components
# tfidf_vectorizer = TfidfVectorizer()
# logistic_regression = LogisticRegression(max_iter=1000)
# random_forest = RandomForestClassifier()
# svm_classifier = SVC()

# # Create pipelines
# pipeline_lr = Pipeline([('tfidf', tfidf_vectorizer), ('clf', logistic_regression)])
# pipeline_rf = Pipeline([('tfidf', tfidf_vectorizer), ('clf', random_forest)])
# pipeline_svm = Pipeline([('tfidf', tfidf_vectorizer), ('clf', svm_classifier)])

# # Define hyperparameters for tuning
# param_grid_lr = {
#     'tfidf__max_features': [5000, 10000, None],
#     'clf__C': [0.1, 1, 10]
# }

# param_grid_rf = {
#     'tfidf__max_features': [5000, 10000, None],
#     'clf__n_estimators': [50, 100, 200]
# }

# param_grid_svm = {
#     'tfidf__max_features': [5000, 10000, None],
#     'clf__C': [0.1, 1, 10],
#     'clf__kernel': ['linear', 'rbf']
# }


# In[72]:


# df.head()


# In[73]:


# df.columns


# In[80]:


# # Perform grid search with cross-validation
# grid_search_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=5, verbose=1)
# grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, verbose=1)
# grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, verbose=1)

# mlflow.sklearn.autolog(max_tuning_runs=None)
# with mlflow.start_run() as run:
#     # Fit the models
#     grid_search_lr.fit(X_train['Review text'], y_train)


# # Auto Logging SVM Experiment Run using MLFlow

# In[91]:


pipe_2 = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('classifier', SVC())
    ]
)


# Observe the Key Value Pair format
parameter_grid_2 = [
    {
        'scaler': [StandardScaler(), MaxAbsScaler()],
        'classifier__kernel' : ['rbf'], 
        'classifier__C' : [0.1, 0.01, 1, 10, 100]
    }, 
    {
        'scaler': [StandardScaler(), MaxAbsScaler()],
        'classifier__kernel' : ['poly'], 
        'classifier__degree' : [2, 3, 4, 5], 
        'classifier__C' : [0.1, 0.01, 1, 10, 100]
    }, 
    {
        'scaler': [StandardScaler(), MaxAbsScaler()],
        'classifier__kernel' : ['linear'], 
        'classifier__C' : [0.1, 0.01, 1, 10, 100]
    }
]


# In[93]:


clf = GridSearchCV(
    estimator=pipe_2, 
    param_grid=parameter_grid_2, 
    scoring='accuracy',
    cv=5,
    return_train_score=True,
    verbose=1
)

# Initialize the auto logger
# max_tuning_runs=None will make sure that all the runs are recorded.
# By default top 5 runs will be recorded for each experiment
mlflow.sklearn.autolog(max_tuning_runs=None)

with mlflow.start_run() as run:
    get_ipython().run_line_magic('time', 'clf.fit(X_train_num, y_train)')


# # Auto Logging All Experiment Runs using MLFlow

# In[101]:


import mlflow

mlflow.set_experiment("flipkart_sentimental_anaysis")


# In[102]:


pipelines = {
    'knn' : Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]), 
    'svc' : Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC())
    ]),
    'logistic_regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ]),
    'random_forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ]),
    'decision_tree': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier())
    ]),
    'naive_bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])
}

# Define parameter grid for each algorithm
param_grids = {
    'knn': [
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__n_neighbors' : [i for i in range(3, 21, 2)], 
            'classifier__p' : [1, 2, 3]
        }
    ],
    'svc': [
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__kernel' : ['rbf'], 
            'classifier__C' : [0.1, 0.01, 1, 10, 100]
        }, 
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__kernel' : ['poly'], 
            'classifier__degree' : [2, 3, 4, 5], 
            'classifier__C' : [0.1, 0.01, 1, 10, 100]
        }, 
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__kernel' : ['linear'], 
            'classifier__C' : [0.1, 0.01, 1, 10, 100]
        }
    ],
    'logistic_regression': [
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__C': [0.1, 1, 10], 
            'classifier__penalty': ['l2']
        }, 
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__C': [0.1, 1, 10], 
            'classifier__penalty': ['l1'], 
            'classifier__solver': ['liblinear']
        }, 
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__C': [0.1, 1, 10], 
            'classifier__penalty': ['elasticnet'], 
            'classifier__l1_ratio': [0.4, 0.5, 0.6],
            'classifier__solver': ['saga']
        }
    ],
    'random_forest': [
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__n_estimators': [50, 100, 200]
        }
    ],
    'decision_tree': [
        {
            'scaler': [StandardScaler(), MaxAbsScaler()],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'naive_bayes': [
        {
            'scaler': [StandardScaler(), MaxAbsScaler()]
        }
    ]
}


# In[103]:


best_models = {}

# Run the Pipeline
for algo in pipelines.keys():
    print("*"*10, algo, "*"*10)
    grid_search = GridSearchCV(estimator=pipelines[algo], 
                               param_grid=param_grids[algo], 
                               cv=5, 
                               scoring='accuracy', 
                               return_train_score=True,
                               verbose=1
                              )
    
    mlflow.sklearn.autolog(max_tuning_runs=None)
    
    with mlflow.start_run() as run:
        get_ipython().run_line_magic('time', 'grid_search.fit(X_train, y_train)')
        
    print('Train Score: ', grid_search.best_score_)
    print('Test Score: ', grid_search.score(X_test, y_test))
    
    best_models[algo] = grid_search.best_estimator_
    print()


# In[100]:


# best_models = {}

# # Run the Pipeline
# for algo in pipelines.keys():
#     print("*"*10, algo, "*"*10)
#     grid_search = GridSearchCV(estimator=pipelines[algo], 
#                                param_grid=param_grids[algo], 
#                                cv=5, 
#                                scoring='accuracy', 
#                                return_train_score=True,
#                                verbose=1
#                               )
    
#     mlflow.sklearn.autolog(max_tuning_runs=None)
    
#     with mlflow.start_run() as run:
#         %time grid_search.fit(X_train_num, y_train)
        
#     print('Train Score: ', grid_search.best_score_)
#     print('Test Score: ', grid_search.score(X_test_num, y_test))
    
#     best_models[algo] = grid_search.best_estimator_
#     print()


# In[ ]:


# Stop the auto logger

# mlflow.sklearn.autolog(disable=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[97]:


import mlflow
mlflow.set_experiment("NLP_Sentiment_Analysis")


# # Defining a memory object to cache Intermediate results

# In[98]:


cachedir = '.cache'
memory = Memory(location=cachedir, verbose=0)

pipelines = {
    'naive_bayes': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', MultinomialNB())
    ], memory=memory),
    'decision_tree': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', DecisionTreeClassifier())
    ], memory=memory),
    'logistic_regression': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', LogisticRegression())
    ], memory=memory)
}

# Define parameter grid for each algorithm
param_grids = {
    'naive_bayes': [
        {
            'vectorization': [CountVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000], 
            'classifier__alpha' : [1, 10]
        }
    ],
    'decision_tree': [
        {
        'vectorization': [CountVectorizer(),TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'logistic_regression': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000], 
            'classifier__C': [0.1, 1, 10], 
            'classifier__penalty': ['elasticnet'], 
            'classifier__l1_ratio': [0.4, 0.5, 0.6],
            'classifier__solver': ['saga'],
            'classifier__class_weight': ['balanced']
        }
    ]
}


# In[99]:


best_models = {}

# Run the Pipeline
for algo in pipelines.keys():
    print("*"*10, algo, "*"*10)
    grid_search = GridSearchCV(estimator=pipelines[algo], 
                               param_grid=param_grids[algo], 
                               cv=5, 
                               scoring='accuracy', 
                               return_train_score=True,
                               verbose=1
                              )
    
    mlflow.sklearn.autolog(max_tuning_runs=None)
    
    with mlflow.start_run() as run:
        get_ipython().run_line_magic('time', 'grid_search.fit(X_train, y_train)')
        
    print('Train Score: ', grid_search.best_score_)
    print('Test Score: ', grid_search.score(X_test, y_test))
    
    best_models[algo] = grid_search.best_estimator_
    print()


# In[ ]:




