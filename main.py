import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import json
import sys


params = json.loads(sys.argv[1])

data = pd.read_csv("dataset/wdbc.data",header=None)
data.head()

features = data.iloc[:,2:].values
label = data.iloc[:,1].values


labelEncoder_Y = LabelEncoder()
label = labelEncoder_Y.fit_transform(label)


trainFeatures, testFeatures, trainLabel, testLabel = train_test_split(features, label, test_size = 0.25, random_state = 0)


sc = StandardScaler()
trainFeatures = sc.fit_transform(trainFeatures)
testFeatures = sc.transform(testFeatures)

classifier = LogisticRegression(**params)
ou = classifier.fit(trainFeatures, trainLabel)
print(ou)

Y_pred = classifier.predict(testFeatures)

cm = confusion_matrix(testLabel, Y_pred)
print(cm)