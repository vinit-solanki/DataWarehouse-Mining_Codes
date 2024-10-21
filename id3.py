import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits)
df = pd.DataFrame(data=digits.data, columns=digits.feature_names)
df['target']=digits.target
print(df.head())

X=df.drop('target',axis=1)
y=df['target']
X_train,y_train,X_test,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
y_preds = model.predict(X_test)
accuracy = accuracy_score(y_preds,y_test)
print("Accuracy:", accuracy)
c_report = classification_report(y_test,y_preds)
print("Classification Report: ",c_report)