<h1>Choosing a Classifier Ensemble</h1>
This Python code is aimed to measure and compare Scikit-Learn Classifier Ensembles.
<h2>Background</h2>
<h3>General Info</h3>
There are many useful classification methods within <a href=https://scikit-learn.org/stable/>Scikit-Learn</a>, machine learning library for Python. In this article, we'll concentrate on <a href=https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble>ensemble methods</a>. Ensemble models combine multiple models to create a better model and produce more accurate predictions.<br><br>
<b>Remember that each dataset has its unique characteristics, meaning the results would vary depending on dataset.</b><br><br>
The aim of this code is to make things <b>easy to decide which ensemble model should be selected</b>. Since the goal of this code is comparison, you can add or remove other models, including non-ensemble ones (like Logistic Regression). If you decide to make changes, the code would still be very similar as long as you use Scikit-Learn library.
<h3>Dataset</h3>
The dataset used here is found on <a href=https://www.kaggle.com/>Kaggle</a> and the information was provided by <a href=https://www.sdss.org/>Sloan Digital Sky Survey</a>. Various measurements about sky objects are presented, and you're asked for classifying the objects as galaxies, quasars or stars. More information about the dataset and each variable can be found <a href=https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey>here</a>. You can download the dataset directly from <a href=https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey/downloads/sloan-digital-sky-survey.zip/2>this link</a>.
<h3>Methodology</h3>
In this article, we'll use Python to compare
<ul>
  <li>AdaBoost</li>
  <li>Bagging</li>
  <li>Extra Trees</li>
  <li>Gradient Boosting</li>
  <li>Random Forest</li>
 </ul>
classifiers. Also, we'll employ <a href=https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>KFold</a> to get average model scores, use the classifiers to make predictions and then create <a href=https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>confusion matrices</a> to visualize prediction results.<br><br>
Let's start.
<h2>Analysis</h2>
<h3>Preparations</h3>
First, we need to import necessary modules.
<pre>
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
</pre>
The reason to import warnings is to suppress future warnings.<br>
We're also going to use the time library to measure the KFold model running times.<br>
<pre>
df = pd.read_csv('C:/Users/Emir/Desktop/Skyserver.csv')
df = df.fillna(0)
df.head()
</pre>
