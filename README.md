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
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>objid</th>
      <th>ra</th>
      <th>dec</th>
      <th>u</th>
      <th>g</th>
      <th>r</th>
      <th>i</th>
      <th>z</th>
      <th>run</th>
      <th>rerun</th>
      <th>camcol</th>
      <th>field</th>
      <th>specobjid</th>
      <th>class</th>
      <th>redshift</th>
      <th>plate</th>
      <th>mjd</th>
      <th>fiberid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.237650e+18</td>
      <td>183.531326</td>
      <td>0.089693</td>
      <td>19.47406</td>
      <td>17.04240</td>
      <td>15.94699</td>
      <td>15.50342</td>
      <td>15.22531</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>267</td>
      <td>3.722360e+18</td>
      <td>STAR</td>
      <td>-0.000009</td>
      <td>3306</td>
      <td>54922</td>
      <td>491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.237650e+18</td>
      <td>183.598371</td>
      <td>0.135285</td>
      <td>18.66280</td>
      <td>17.21449</td>
      <td>16.67637</td>
      <td>16.48922</td>
      <td>16.39150</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>267</td>
      <td>3.638140e+17</td>
      <td>STAR</td>
      <td>-0.000055</td>
      <td>323</td>
      <td>51615</td>
      <td>541</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.237650e+18</td>
      <td>183.680207</td>
      <td>0.126185</td>
      <td>19.38298</td>
      <td>18.19169</td>
      <td>17.47428</td>
      <td>17.08732</td>
      <td>16.80125</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>268</td>
      <td>3.232740e+17</td>
      <td>GALAXY</td>
      <td>0.123111</td>
      <td>287</td>
      <td>52023</td>
      <td>513</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.237650e+18</td>
      <td>183.870529</td>
      <td>0.049911</td>
      <td>17.76536</td>
      <td>16.60272</td>
      <td>16.16116</td>
      <td>15.98233</td>
      <td>15.90438</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>269</td>
      <td>3.722370e+18</td>
      <td>STAR</td>
      <td>-0.000111</td>
      <td>3306</td>
      <td>54922</td>
      <td>510</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.237650e+18</td>
      <td>183.883288</td>
      <td>0.102557</td>
      <td>17.55025</td>
      <td>16.26342</td>
      <td>16.43869</td>
      <td>16.55492</td>
      <td>16.61326</td>
      <td>752</td>
      <td>301</td>
      <td>4</td>
      <td>269</td>
      <td>3.722370e+18</td>
      <td>STAR</td>
      <td>0.000590</td>
      <td>3306</td>
      <td>54922</td>
      <td>512</td>
    </tr>
  </tbody>
</table>
