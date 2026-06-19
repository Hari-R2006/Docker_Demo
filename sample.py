import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
#x hours studied
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])

#y marks
y = np.array([35, 42, 49, 58, 60, 68, 74, 80, 89, 95, 98, 100])

#train linear regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

#train SVR model
svr_model = SVR(kernel='rbf',C=1000, gamma=0.1)
svr_model.fit(X, y)

#save the model
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('svr_model.pkl', 'wb') as f:
    pickle.dump(svr_model, f)

print("Models trained and saved successfully!")
