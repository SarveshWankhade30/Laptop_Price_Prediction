import pandas as pd
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# 1. Load data
data = pd.read_csv("laptop_data.csv")

X = data.drop("Price", axis=1)
y = data["Price"]

# 2. Preprocessing
categorical = X.select_dtypes(include="object").columns.tolist()
numerical = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])

# 3. Models
models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "SVR": SVR(),
    "DecisionTree": DecisionTreeRegressor()
}

params = {
    "Linear": {},
    "RandomForest": {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5]
    },
    "SVR": {
        "model__kernel": ['rbf'],
        "model__C": [1, 10]
    },
    "DecisionTree": {
        "model__max_depth": [None, 5, 10]
    }
}

best_score = -1
best_model = None

# 4. Train and evaluate all models
for name in models:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", models[name])
    ])
    clf = GridSearchCV(pipeline, params[name], cv=3, scoring='r2')
    clf.fit(X, y)
    print(f"{name} R2 Score: {clf.best_score_}")

    if clf.best_score_ > best_score:
        best_score = clf.best_score_
        best_model = clf.best_estimator_

# 5. Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("✅ Best model saved as model.pkl")
