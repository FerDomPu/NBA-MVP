import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Impport data
stats = pd.read_csv('player_mvp_stats.csv')
del stats['Unnamed: 0']

# Display all columns
pd.set_option('display.max_columns', None)

# Check null values
stats.isnull().sum()
stats = stats.fillna(0)
# stats.to_csv('player_mvp_stats_2.csv')

# Check values
stats.describe()
stats.dtypes

# Correlation matrix: Look for correlation > 0.7
stats_num = stats.select_dtypes(exclude='object')
corr_matrix = stats_num.corr()
# sns.heatmap(corr_matrix,annot=True,cmap='RdBu_r')
# plt.show()
# for i in range(len(corr_matrix.columns)):
#     for j in range(i):
#         # Print variables with high correlation
#         if abs(corr_matrix.iloc[i, j]) > 0.7:
#             print(corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])

stats = stats.sort_values(by=['Year','Player'], ignore_index=True)
            
## Features and target value
# Players:  Partidos jugados, Minutos por partido, Porcentajes de tiro, Rebotes totales, Resto de estadisticas
# Teams: W/L%
# MVP: Share (target value)
features = ['G','MP', 'FG%', '3P%', '2P%', 'FT%', 'TRB', 'AST', 'STL', 'BLK', 'TOV','PTS', 'W/L%']
target = ['Share']

## MaxMin Normalization para tener los maximos y minimos de cada temporada
years = list(stats.Year.unique())

stats_normalized_list = []
for year in years:
    scaler = MinMaxScaler()
    stats_year_normalized = scaler.fit_transform(stats[stats.Year == year][features + target])
    stats_year_normalized = pd.DataFrame(stats_year_normalized, columns=features+target,index=stats[stats.Year == year][features + target].index)
    stats_normalized_list.append(stats_year_normalized)

stats_model = pd.concat(stats_normalized_list)

stats_normalized = pd.concat([stats[['Player', 'Year', 'Team']],stats_model],axis=1)

## Probar un modelo de Linear Regression con el último año
X_train = stats_normalized[stats_normalized.Year < 2022][features]
y_train = stats_normalized[stats_normalized.Year < 2022][target]
X_test = stats_normalized[stats_normalized.Year == 2022][features]
y_test = stats_normalized[stats_normalized.Year == 2022][target]

reg = LinearRegression()
reg.fit(X_train,y_train)

y_predict = reg.predict(X_test)

y_predict = pd.DataFrame(y_predict,columns=['predictions'],index=X_test.index)
prediction_2022 = pd.concat([stats_normalized[stats_normalized.Year == 2022][['Player','Share']],y_predict],axis=1)
prediction_2022.sort_values('Share',ascending=False).head(20)

## Funciones para analizar los resultados
def add_ranks(predictions):
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1,predictions.shape[0]+1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1,predictions.shape[0]+1))
    predictions["Diff"] = (predictions["Rk"] - predictions["Predicted_Rk"])
    return predictions

def find_ap(combination):
    actual = combination.sort_values("Share", ascending=False).head(5)
    predicted = combination.sort_values("predictions", ascending=False)
    ps = []
    found = 0
    seen = 1
    for index,row in predicted.iterrows():
        if row["Player"] in actual["Player"].values:
            found += 1
            ps.append(found / seen)
        seen += 1
    return sum(ps) / len(ps)

def backtest(stats, model, years, features):
    aps = []
    all_predictions = []
    for year in years:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        model.fit(train[features],train['Share'])
        predictions = model.predict(test[features])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player",'Year', "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)

model = LinearRegression()

mean_ap, aps, all_predictions = backtest(stats_normalized, model, years[5:], features)

## Hyperparameter tuning
# from sklearn.linear_model import RidgeCV

# model = RidgeCV(alphas=np.logspace(-2,1))
# model.fit(X_train,y_train)
# y_predict = model.predict(X_test)
# y_predict = pd.DataFrame(y_predict,columns=['predictions'],index=X_test.index)
# prediction_2022 = pd.concat([stats_normalized[stats_normalized.Year == 2022][['Player','Share']],y_predict],axis=1)
# prediction_2022.sort_values('Share',ascending=False).head(20)
# prediction_2022 = add_ranks(prediction_2022)
# find_ap(prediction_2022)

## Random Forest Regressor
# from sklearn.ensemble import RandomForestRegressor

# model = RandomForestRegressor()
# model.fit(X_train,y_train)
# y_predict = model.predict(X_test)
# y_predict = pd.DataFrame(y_predict,columns=['predictions'],index=X_test.index)
# prediction_2022 = pd.concat([stats_normalized[stats_normalized.Year == 2022][['Player','Share']],y_predict],axis=1)
# prediction_2022.sort_values('Share',ascending=False).head(20)
# prediction_2022 = add_ranks(prediction_2022)
# find_ap(prediction_2022)

