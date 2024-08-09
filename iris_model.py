from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

iris = datasets.load_iris()

X = iris.data
y = iris.target


x_train, x_test, y_train, y_test = train_test_split(X, y)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_m = SVC()

lin_regr = lin_reg.fit(x_test, y_test)
log_regr = lin_reg.fit(x_test, y_test)
svc_mo = svc_m.fit(x_test, y_test)


with open ('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)


with open ('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open ('svc_m.pkl', 'wb') as svc:
    pickle.dump(svc_mo, svc)