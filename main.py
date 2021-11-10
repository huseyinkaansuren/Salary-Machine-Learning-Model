import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

dataFrame = pd.read_csv("Salary_Data.csv")
# print(dataFrame.isnull().sum())

# sb.displot(dataFrame["Salary"])
# plt.show()

# sb.scatterplot(x="YearsExperience", y="Salary", data=dataFrame)
# plt.show()

y = dataFrame["Salary"].values
x = dataFrame.drop("Salary", axis=1).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()

model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))
model.add(Dense(10, activation="relu"))


model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), batch_size=5, epochs=550)

lossData = pd.DataFrame(model.history.history)

predictArray = model.predict(x_test)

print(mean_absolute_error(y_test, predictArray))
print(dataFrame.describe())

print(dataFrame.iloc[2])
newPerson = dataFrame.drop("Salary",axis=1).iloc[2]
newPerson = scaler.transform(newPerson.values.reshape(-1,1))
print(model.predict(newPerson))