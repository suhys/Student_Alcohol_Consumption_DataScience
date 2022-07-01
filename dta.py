import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

file_path = "student-mat.csv"

math_data = pd.read_csv(file_path)

print(math_data)

y = math_data.G3

print(y.head)

math_features = ['studytime', 'failures', 'freetime', 'Dalc', 'Walc', 'absences', 'G1', 'G2']

X = math_data[math_features]

print(X.head())

math_model = DecisionTreeRegressor(random_state=1)

math_model.fit(X, y)

print("Making predictions for the following ")
print(X.head())
print("The predtions are")
print(math_model.predict(X.head()))

predicted_final_grade = math_model.predict(X)
print(mean_absolute_error(y, predicted_final_grade))