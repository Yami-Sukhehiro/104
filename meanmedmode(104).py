from pandas.core.common import random_state
from google.colab import files
data_to_upload = files.upload()
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn.tree import export_graphviz
from sklearn.naive_bayes import GaussianNB
import statistics

df  =  pd.read_csv("SOCR-HeightWeight.csv")
weightlist = df["Weight(Pounds)"].tolist()
heightlist = df["Height(Inches)"].tolist()
meanh = statistics.mean(weightlist)
meanw = statistics.mean(heightlist)

meanhm = statistics.median(weightlist)
meanwm = statistics.median(heightlist)

meanhmm = statistics.median(weightlist)
meanwmm = statistics.median(heightlist)

stdhmm = statistics.stdev(weightlist)
stdwmm = statistics.stdev(heightlist)

print(meanh, meanw, meanhm, meanwm, meanhmm, meanwmm)