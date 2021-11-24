import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def print_r(name, y_t, y_r):
    print(name)
    print("accuracy (доля правильных ответов): ", accuracy_score(y_t, y_r))
    print("precision (полнота): ", precision_score(y_t, y_r))
    print("recall (точность): ", recall_score(y_t, y_r))
    print("f1 (объединенная): ", f1_score(y_t, y_r))


df = pd.read_csv('Customer_Churn.csv')

column_X = ['gender', 'SeniorCitizen', 'PhoneService', 'MultipleLines', 'InternetService', 'Partner', 'Dependents',
            'PaymentMethod']
column_Y = ['Churn']

X = pd.DataFrame(columns=column_X)
X.gender = df.gender.astype("category").cat.codes
X.SeniorCitizen = df.SeniorCitizen.astype("category").cat.codes
X.PhoneService = df.PhoneService.astype("category").cat.codes
X.MultipleLines = df.MultipleLines.astype("category").cat.codes
X.InternetService = df.InternetService.astype("category").cat.codes
X.Partner = df.Partner.astype("category").cat.codes
X.Dependents = df.Dependents.astype("category").cat.codes
X.PaymentMethod = df.PaymentMethod.astype("category").cat.codes

Y = pd.DataFrame(columns=column_Y)
Y.Churn = df.Churn.astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

bayes = MultinomialNB()
bayes.fit(X_train, y_train)
y_bayes = bayes.predict(X_test)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_tree = tree.predict(X_test)

neural = MLPClassifier()
neural.fit(X_train, y_train)
y_neural = neural.predict(X_test)

print_r("MultinomialNB", y_test, y_bayes)
print_r("DecisionTreeClassifier", y_test, y_tree)
print_r("MLPClassifier", y_test, y_neural)
