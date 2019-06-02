<h1>Choosing a Classifier Ensemble</h1>
This Python code is aimed to measure and compare predictive powers of Scikit-Learn Classifier Ensembles.
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
classifiers. All classifiers will be used with their default parameters.<br><br>
Also, we'll employ <a href=https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html>KFold</a> to get average model scores, use the classifiers to make predictions and then create <a href=https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html>confusion matrices</a> to visualize prediction results.<br><br>
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
Now that the data is properly loaded, let's take a deper look into each object class.
<pre>
df.groupby(['class']).mean()
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
      <th>redshift</th>
      <th>plate</th>
      <th>mjd</th>
      <th>fiberid</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GALAXY</th>
      <td>1.237650e+18</td>
      <td>177.333570</td>
      <td>15.764372</td>
      <td>18.804339</td>
      <td>17.350216</td>
      <td>16.649506</td>
      <td>16.272770</td>
      <td>16.017923</td>
      <td>996.711685</td>
      <td>301.0</td>
      <td>3.654662</td>
      <td>300.963585</td>
      <td>5.379141e+17</td>
      <td>0.080325</td>
      <td>477.680672</td>
      <td>52030.280912</td>
      <td>340.108844</td>
    </tr>
    <tr>
      <th>QSO</th>
      <td>1.237650e+18</td>
      <td>177.468000</td>
      <td>20.570639</td>
      <td>18.942928</td>
      <td>18.678714</td>
      <td>18.498535</td>
      <td>18.360007</td>
      <td>18.274761</td>
      <td>1036.120000</td>
      <td>301.0</td>
      <td>3.694118</td>
      <td>304.983529</td>
      <td>1.447231e+18</td>
      <td>1.218366</td>
      <td>1285.305882</td>
      <td>52694.289412</td>
      <td>381.558824</td>
    </tr>
    <tr>
      <th>STAR</th>
      <td>1.237650e+18</td>
      <td>172.962158</td>
      <td>12.544824</td>
      <td>18.330439</td>
      <td>17.130547</td>
      <td>16.732093</td>
      <td>16.594047</td>
      <td>16.531119</td>
      <td>950.886561</td>
      <td>301.0</td>
      <td>3.632225</td>
      <td>303.552264</td>
      <td>3.018202e+18</td>
      <td>0.000043</td>
      <td>2680.613198</td>
      <td>54093.892823</td>
      <td>362.838391</td>
    </tr>
  </tbody>
</table>
It is already obvious that any variable that contains some sort of ID number won't help for analysis; so 'objid' and 'specobjid' will be out. Beside those two, we can also see that the variable 'rerun' is constant among variables; is doesn't provide any useful information either.<br><br>
So, we get our predictors and target like this:
<pre>
x = df.drop(['class','objid','rerun','specobjid'], axis=1)
y = df.filter(['class'])
</pre>
Next, we need to split the dataset to train the models and test them on the splits.
<pre>
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)
</pre>
<h3>Method Declaration</h3>
Now it's time to specify the models to be compared and the methods of comparison.
<pre>
models = []
models.append(('ABC', AdaBoostClassifier()))
models.append(('BC ', BaggingClassifier()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('RFC', RandomForestClassifier()))
</pre>
We're using a list for all models. First value of each object are the initials of model names. There's one extra space character at the end of BC, that is for alignment.<br>
This also is the part you can change if you want to make similar classification comparisons. Just append any model that you've imported into this list (remember to use that model's own initials!).
<pre>
kfold = model_selection.KFold(n_splits=10, random_state=7)
scoring = 'accuracy'
</pre>
KFold will split the training data into 10 pieces, and use each one as validation while the other 9 pieces serve as training data.<br>
Models will be scored according to their accuracy.
<h3>Models Runs and Scores</h3>
It's time to see what the models can produce.
<pre>
for name, model in models:
    start_time = time.time()
    cv_results = model_selection.cross_val_score(model, x, y.values.ravel(), cv=kfold, scoring=scoring)
    print('%s: %f  in %.2f seconds' % (name, cv_results.mean(), time.time() - start_time))
</pre>
This part of the code will print the name, accuracy and time to complete the KFold validation for each model.<br>
The start_time will fetch the exact time then the for loop starts the next loop. When it's time to print the result, that time will be subtracted from the current time, and '%.2f' of the code will print it as a float with 2 decimal places.
<br><br>
Here are the results:
<pre>
ABC: 0.843600  in 8.53 seconds
BC : 0.988700  in 4.99 seconds
ETC: 0.970200  in 0.54 seconds
GBC: 0.989700  in 22.41 seconds
RFC: 0.987900  in 1.31 seconds
</pre>
Gradient Boosting Classifier has the best score, but also took the longest time to run. Bagging Classifier is a close second with much better running time.<br><br>
Let's use the models for predictions:
<pre>
y_pred = []
for name, model in models:
    model.fit(x_train,y_train.values.ravel())
    PredictionResults = model.predict(x_test)
    y_pred.append([name, PredictionResults])
    print('%s fitted and used for predictions.' % name)
</pre>
We have a list called y_pred, and it will save all the prediction information for each model. Notice that the name of the model is also recorded into the list because we will need it during the cretion of confusion matrices.<br><br>
The data is fitted into all models, prediction results are individually saved into the 'PredictionResults' and then appended to y_pred for future use. <b>Because this is a loop, anything you want to use later must be recorded into a variable that is out of the loop!</b><br><br>
The print command is here to show that the code ran succesfully. 
<pre>
ABC fitted and used for predictions.
BC  fitted and used for predictions.
ETC fitted and used for predictions.
GBC fitted and used for predictions.
RFC fitted and used for predictions.
</pre>
<h3>Visualization</h3>
So, we got our predictions. But how do we see and compare them?<br><br>
Here we'll use confusion matrices. Below is a custom function for creating them; this was taken from <a href =https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py>an example on Scikit-Learn website</a> and several changes were made to give it a little bit of a custom look.
<pre>
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
</pre>
This code can be used for both standard and normalized confusion matrices. It has a default title, and the color map can be specified. Details of this code won't be discussed here since it's out of the scope of this article.<br><br>
In the final step, we'll run two confusion matrices for each prediction set; one will be standard, one will be normalized. We'll use another for loop to do this.
<pre>
for name, pred in y_pred:
    conmat = confusion_matrix(y_test, pred)
    y_names=np.unique(y_test.iloc[:,0:1].values)
    f = plt.figure(figsize=(12, 5))
    f.add_subplot(1,2,1)
    plot_confusion_matrix(conmat, classes=y_names, title='%s Standard Confusion Matrix' % name)
    f.add_subplot(1,2,2)
    plot_confusion_matrix(conmat, classes=y_names, normalize=True, title='%s Normalized Confusion Matrix' % name)
    plt.subplots_adjust(wspace=0.2)
    plt.show()
</pre>
All our predictions in our y_pred list go through the confusion_matrix() function. Names of the target variables are taken from the y_test dataset. For each confusion matrix pair, the one at the left will be the standard plot and the one at the left will be the normalized plot. The plots will have a space between them that is 20% of the whole axis width.<br><br>
All confusion matrices are shown below:<br>
<img src="https://github.com/EmirKorkutUnal/Choosing-a-Classifier-Ensemble/blob/master/images/ABCConMat.jpg">
<img src="https://github.com/EmirKorkutUnal/Choosing-a-Classifier-Ensemble/blob/master/images/BCConMat.jpg">
<img src="https://github.com/EmirKorkutUnal/Choosing-a-Classifier-Ensemble/blob/master/images/ETCConMat.jpg">
<img src="https://github.com/EmirKorkutUnal/Choosing-a-Classifier-Ensemble/blob/master/images/GBCConMat.jpg">
<img src="https://github.com/EmirKorkutUnal/Choosing-a-Classifier-Ensemble/blob/master/images/RFCConMat.jpg">
As you can see, confusion matrices allow us to easily notice where the models were successful and where they failed.<br><br>
<b>AdaBoost Classifier had a big problem labeling stars</b>; it labeled 78% of all stars within the test dataset as galaxies.<br>
<b>Bagging Classifier was quite successful</b>, its only notable fault was labeling 6% of quasars as galaxies.<br>
<b>Extra Trees Classifier performed well</b>, though had problems both in quasar and star labeling.<br>
<b>Gradient Boosting Classifier has the best results</b>, maximum partial error rate is 5%.<br>
<b>Random Forest Classifier accomplished good results</b>, again having difficulties labeling quasars.
<h3>Futrher Interpretation for this Analysis</h3>
If your only choice of classifying these objects would be one of these 5 models, you would have to go with the Gradient Boosting Classifier despite the 5% partial error rate - which is not that high and also consider that the part where the error is made is relatively small. In real world, this might not be the case, so you might go one of the following directions:
<ul>
  <li><b>Play around with default parameters of each ensemble model</b> until you get a better result. Keep in mind that the both <b>AdaBoost and Bagging Classifiers have a base_estimator parameter</b> where you can change the standard decision tree estimator into something else, like Logistic Regression.</li> 
  <li>Look at alternative models, such as Artifical Neural Network Classifier, and measure their accuracies on the dataset.</li>
  <li>All 5 models had some level of problem when it came to labeling quasars; this indicated that the dataset itself may not contain sufficient information to classify all objects correctly. You may want to search for other related variables to increase model accuracy.</li>
</ul>
<h2>Conclusion</h2>
Although AdaBoost Classifier didn't perform as well in this example, <b>you can use all of these ensembles for your classification analyzes. They are powerful methods and can increase the accuracy of any existing simple method.<b> 
