#@title Run this to Import Data and Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
#%matplotlib inline

from urllib.request import urlretrieve
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn import  metrics
from sklearn import tree
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from scipy.signal import savgol_filter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay,precision_score,recall_score,f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize

import warnings
warnings.filterwarnings('ignore')

def analyze_results(model, X_train, y_train, X_test, y_test):
    """
    Helper function to help interpret and model performance.

    Args:
    model: estimator instance
    X_train: {array-like, sparse matrix} of shape (n_samples, n_features)
    Input values for model training.
    y_train : array-like of shape (n_samples,)
    Target values for model training.
    X_test: {array-like, sparse matrix} of shape (n_samples, n_features)
    Input values for model testing.
    y_test : array-like of shape (n_samples,)
    Target values for model testing.

    Returns:
    None
    """
    print("-------------------------------------------")
    print("Model Results: ")
    print("")
    print("The first graph is the training data accuracy:")
    ConfusionMatrixDisplay.from_estimator(model, X_train, y_train)
    plt.show()
    print("The second graph is the testing data accuracy:")
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.show()

def check_norm_answers(v1, v2, v3, v4):
  print("Value 1:", v1 == 0.14)
  print("Value 2:", v2 == 0)
  print("Value 3:", v3 == 0.86)
  print("Value 4:", v4 == 1.0)

df_train = pd.read_csv('exoTrain.csv')
df_train['LABEL'] = df_train['LABEL'] - 1
df_test = pd.read_csv('exoTest.csv')
df_test['LABEL'] = df_test['LABEL'] - 1

print("Now, let's approach this as a classification task: based on our flux data (the light value associated with each time stamp), does each data point represent an exoplanet star or a non-exoplanet star?")
X_train = df_train.drop('LABEL', axis=1)
y_train = df_train['LABEL']

X_test = df_test.drop('LABEL', axis=1)
y_test = df_test['LABEL']

# Step 1: Create our model.. Let's start by using a KNeighborsClassifier model, which has already been imported from Scikit-learn.
print("To do this, we will use a classifier model, KNN (K-Nearest-Neighbors). \nTo put it simply, this type of model classifies a value by finding the \"k\" nearest points in the dataset and using majority rule.\n")
model = KNeighborsClassifier(n_neighbors = 5)

# Step 2: Train our model  Now train the KNN model defined on X_train and y_train using the fit method.
model.fit(X_train, y_train)

# Step 3: Predictions and Accuracy
# Calculate the predictions and accuracies on X_train and X_test using our trained model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
print("Training data accuracy: ", accuracy_score(y_train, train_predictions))
print("Testing data accuracy: ", accuracy_score(y_test, test_predictions)) 

# Step 4: Confusion Matrices
print("\nConfusion Matrices help visualize the model's accuracy, which can help see the kinds of biases it might have - even if it does has a high overall accuracy")
# The code below uses the analyze_results function, which will display these confusion matrices.
analyze_results(model, X_train, y_train, X_test, y_test)
print("\nAs you can see, even though the model is predicting the non-exoplanets correctly, it cannot predict the true exoplanets and is giving many false negatives.")