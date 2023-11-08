#--- Import library ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder # categorical variables
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor # model
from sklearn.linear_model import Ridge,Lasso,ElasticNet,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import sklearn
import plotly.express as px # plots
import plotly.graph_objects as go

# warnings.filterwarnings('ignore')

# --- Read dataset ---
df = pd.read_csv("./data/insurance.csv")
df

# --- summary of variables
df.describe()

# --- data cleaning/wrangling ---

# --- check for duplicates ---
duplicates = df.duplicated().sum()
print(f"Total duplicates: {duplicates}")

# --- drop duplicate ---
df.drop_duplicates(inplace=True)
df

# --- Check for NA ---
null_values = df.isna().sum()
print(f"No missing values:\n{null_values}")

# --- Exploratory data analysis ---
fig = px.histogram(df, x = 'charges', title="Distribution of Charges")
fig.update_xaxes(title_text="Charges")
fig.show()

# df.columns

fig = px.scatter(df, x="age", y="charges", color="smoker")
fig.update_yaxes(title_text="Charges")
fig.update_xaxes(title_text="Age")
fig.update_layout(title_text="Scatter Plot of Age vs. Charges by Smoking Status")
fig.show()

fig = px.scatter(df,x="age", y="charges", color="smoker", facet_col="region")
fig.update_layout(title_text="Scatter Plot of Age vs. Charges by Smoking Status (Facetted by Region)")
fig.show()

fig = px.scatter(df,x="age", y="charges", color="smoker", facet_col="region", facet_row="sex")
fig.update_layout(title_text="Scatter Plot of Age vs. Charges by Smoking Status (Facetted by Region and Sex)")
fig.show()

fig = px.box(df, y="charges", x="children",
             title="Boxplot of Charges by number of children")
fig.show()

corr_matrix = df.corr(numeric_only=True)
corr_matrix
fig = go.Figure(go.Heatmap(
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    z=corr_matrix.values, colorscale='Viridis'))
fig.update_layout(title="Correlation Heatmap of Numeric Features")
fig.show()


#--- Preprocessing Label Encoding of Categorical Feature ---
lab_encode = LabelEncoder()

for var in ["sex","smoker"]:
    df[var] = lab_encode.fit_transform(df[var])


# --- OneHot Encoded ---
one_hot_encode = pd.get_dummies(df["region"])
one_hot_encode


# --- Combined data ---
df1 = pd.concat([df, one_hot_encode], axis=1)
df1.columns

# --- drop varaibles "region" ---
df1.drop(columns=["region"], axis=1, inplace=True)
df1

# --- model predictions ---
X = df1.drop(columns=["charges"], axis=1)
y = df1["charges"]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# --- Random forest model ---
random_forest_model = RandomForestRegressor(n_estimators=50, n_jobs=2,  random_state=42)

# --- cross validation on training set
sklearn.metrics.get_scorer_names() 

cross_score = cross_val_score(random_forest_model,
                X_train,
                y_train,
                scoring="neg_mean_absolute_error",
                cv=10)

std = np.std(np.sqrt(np.abs(cross_score)))
std

# --- fit random forest model ---
random_forest_model.fit(X_train, y_train)

# --- prediction of model
prediciton = random_forest_model.predict(X_test)
pred_10 = np.round(prediciton[:10], 2)
y_test_10 = np.array(np.round(y_test[:10], 2))

# --- results
data = {
    "Actual Charges": y_test_10,
    "Predicted Charges": pred_10 
}

compare = pd.DataFrame(data)
compare

# --- Results on test set ---
mse = mean_squared_error(y_test, prediciton)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, prediciton)

print(f'Root Mean Squared Error (RMSE) {rmse:.2f}')
print(f'Mean Absolute Error (RMSE) {mae:.2f}')

results = {
    "Actual": np.array(y_test).round(2),
    "Prediction":prediciton.round(2)
}

results_df = pd.DataFrame(results)
results_df

fig = px.scatter(results_df, x="Prediction", y="Actual")
fig.add_shape(type="line",
             x0=results_df["Prediction"].min(),
             y0=results_df["Prediction"].min(),
             x1=results_df["Prediction"].max(),
             y1=results_df["Prediction"].max(),
             line=dict(color="red", width=2))
fig.update_layout(title_text="Actual vs Prediction")
fig.show()

# --- Running different models ---

""" 
# --- read R train, valdiation and test datasets ---
train_set = pd.read_csv("./train_test_dataset/train.csv")
train_set.drop([train_set.columns[0]], axis=1, inplace=True)
validation_set = pd.read_csv("./train_test_dataset/test.csv")
validation_set.drop([validation_set.columns[0]], axis=1, inplace=True)
test_set = pd.read_csv("./train_test_dataset/test.csv")
test_set.drop([test_set.columns[0]], axis=1, inplace=True)

X_train = train_set.drop(columns=['charges'], axis=1)
y_train = train_set['charges']

X_validation = validation_set.drop(columns=['charges'], axis=1)
y_validation = validation_set['charges']

X_test = test_set.drop(columns=['charges'], axis=1)
y_test = test_set['charges']
"""

# --- models ---
models = [
    ("Ridge",Ridge()),
    ("Lasso", Lasso()),
    ("ElasticNet", ElasticNet()),
    ("Linear Regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor()),
    ("AdaBoost", AdaBoostRegressor()),
    ("Gradient Boosting", GradientBoostingRegressor()),
    ("Random Forest", RandomForestRegressor()),
    ("KNN", KNeighborsRegressor()),
    ("Neural Network", MLPRegressor(max_iter=10000, learning_rate_init=0.001)),
    ("SVM", SVR())
]

# --- directionary for scores ---
model_scores = {}

# --- Cross-validation ---
custom_cv = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=custom_cv, scoring="neg_mean_absolute_error", n_jobs=4)
    # stores absolute error
    model_scores[model_name] = -scores.mean()

# --- Results: cross validation ---
for model_name, score in model_scores.items():
    print(f"{model_name}: Cross-validation MAE = {score:.2f}")

pd.DataFrame(list(model_scores.items()), columns=['Model','Cross-Validation: MAE']).round(2)


# --- Select best model ---
best_model_name = min(model_scores, key=model_scores.get)
print(f"Best model {best_model_name}")
best_model = [model for model_name, model in models if model_name == best_model_name][0]

# --- fit best model ---
best_model.fit(X_train, y_train)

# --- Predictions ---
predictions = best_model.predict(X_test)

# --- final model results ---
mae = mean_absolute_error(y_test.round(2), prediciton.round(2)).round(2)
mse = mean_squared_error(y_test.round(2), prediciton.round(2)).round(2)
rmse = np.sqrt(mse).round(2)

results = {
    "Model": [best_model_name],
    "Mean Absolute Error": [mae],
    "Root Mean Square Error": [rmse]
}

pd.DataFrame(results)
