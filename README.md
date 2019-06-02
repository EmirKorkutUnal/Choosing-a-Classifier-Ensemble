<h1>Choosing a Classifier Ensemble</h1>
This Python code is aimed to measure and compare Scikit-Learn Classifier Ensembles.
<h2>Background</h2>
<h3>General Info</h3>
There are many useful classification methods within <a href=https://scikit-learn.org/stable/>Scikit-Learn</a>, machine learning library for Python. In this article, we'll concentrate on <a href=https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble>ensemble methods</a>. Ensemble models combine multiple models to create a better model and produce more accurate predictions.<br><br>
<b>Remember that each dataset has its unique characteristics, meaning the results would vary depending on dataset.</b><br><br>
The aim of this code is to make things <b>easy to decide which ensemble model should be selected</b>. Since the goal of this code is comparison, you can add or remove otyher models, including non-ensemble ones (like Logistic Regression). If you decide to make changes, the code would still be very similar as long as you use Scikit-Learn library.
<h3>Dataset</h3>
The dataset used here is found on <a href=https://www.kaggle.com/>Kaggle</a>. It is a database about sky objects, and the information was provided by <a href=https://www.sdss.org/>Sloan Digital Sky Survey</a>. Various measurements about sky objects are presented, and you're asked for classifying the objects as galaxies, quasars or stars. More information about each variable can be found <a href=https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey>here</a>. You can download the dataset directly from <a href=https://www.kaggle.com/lucidlenn/sloan-digital-sky-survey/downloads/sloan-digital-sky-survey.zip/2>this link</a>.

