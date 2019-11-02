# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df=pd.read_csv(path)
df.head()
X=df.drop('Price',axis=1)
y=df['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)
corr=X_train.corr()
print(corr)

#Code starts here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
r2=regressor.score(X_test,y_test)
print(r2)
# Code starts here


# --------------
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso.fit(X_train,y_train)
lasso_pred=lasso.predict(X_test)
r2_lasso=lasso.score(X_test,y_test)
print(r2_lasso)
# Code starts here



# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge=Ridge()
ridge.fit(X_train,y_train)
ridge_pred=ridge.predict(X_test)
r2_ridge=ridge.score(X_test,y_test)
print(r2_ridge)


# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
regressor=LinearRegression()
score=cross_val_score(regressor,X_train,y_train,cv=10)
mean_score=score.mean()
print(mean_score)
#Code starts here


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
model=make_pipeline(PolynomialFeatures(2),LinearRegression())
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r2_poly=model.score(X_test,y_test)
print(r2_poly)
#Code starts here


