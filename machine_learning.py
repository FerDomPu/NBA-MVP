# Import
import pandas as pd
from sklearn.linear_model import Ridge

# Import data
stats = pd.read_csv('player_mvp_stats.csv')
del stats['Unnamed: 0']
index_drop = len(stats) - 7
stats.drop(stats.index[index_drop:],inplace=True)

# Check null values
stats.isnull().sum()
stats = stats.fillna(0)

## Training a machine learning model
# Look at the columns
stats.columns
predictors = ['Age', 'G', 'GS', 'MP', 'FG', 'FGA',
              'FG%', '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT',
              'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 
              'PF', 'PTS', 'Year', 'W', 'L', 'W/L%', 'GB', 'PS/G', 'PA/G', 'SRS']

train = stats[stats['Year'] < 2022]
test = stats[stats['Year'] == 2022]

# Create a ridge regression model
reg = Ridge(alpha=.1)

# Fit the model with training data
reg.fit(train[predictors], train['Share'])

# Predictions
predictions = reg.predict(test[predictors])
predictions = pd.DataFrame(predictions, columns=['predictions'], index=test.index)
combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
combination.sort_values("Share", ascending=False).head(20)


# Take metrics
from sklearn.metrics import mean_squared_error

mean_squared_error(combination["Share"], combination["predictions"])

actual = combination.sort_values("Share", ascending=False)
predicted = combination.sort_values("predictions", ascending=False)
actual["Rk"] = list(range(1,actual.shape[0]+1))
predicted["Predicted_Rk"] = list(range(1,predicted.shape[0]+1))
actual.merge(predicted, on="Player").head(5)

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

ap = find_ap(combination)
ap

years = list(range(1990,2023))

aps = []
all_predictions = []

for year in years[5:]:
    train = stats[stats["Year"] < year]
    test = stats[stats["Year"] == year]
    reg.fit(train[predictors],train["Share"])
    predictions = reg.predict(test[predictors])
    predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
    combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
    all_predictions.append(combination)
    aps.append(find_ap(combination))
    
def add_ranks(predictions):
    predictions = predictions.sort_values("predictions", ascending=False)
    predictions["Predicted_Rk"] = list(range(1,predictions.shape[0]+1))
    predictions = predictions.sort_values("Share", ascending=False)
    predictions["Rk"] = list(range(1,predictions.shape[0]+1))
    predictions["Diff"] = (predictions["Rk"] - predictions["Predicted_Rk"])
    return predictions

add_ranks(all_predictions[1])

pd.concat([pd.Series(reg.coef_), pd.Series(predictors)], axis=1).sort_values(0, ascending=False)

def backtest(stats, model, years, predictors):
    aps = []
    all_predictions = []
    for year in years:
        train = stats[stats["Year"] < year]
        test = stats[stats["Year"] == year]
        model.fit(train[predictors],train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)

mean_ap, aps, all_predictions = backtest(stats, reg, years[5:], predictors)
mean_ap


# Random forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=1, min_samples_split=5)

mean_ap, aps, all_predictions = backtest(stats, rf, years[28:], predictors)

mean_ap

# Standardizing our data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

def backtest(stats, model, years, predictors):
    aps = []
    all_predictions = []
    for year in years:
        train = stats[stats["Year"] < year].copy()
        test = stats[stats["Year"] == year].copy()
        sc.fit(train[predictors])
        train[predictors] = sc.transform(train[predictors])
        test[predictors] = sc.transform(test[predictors])
        model.fit(train[predictors],train["Share"])
        predictions = model.predict(test[predictors])
        predictions = pd.DataFrame(predictions, columns=["predictions"], index=test.index)
        combination = pd.concat([test[["Player", "Share"]], predictions], axis=1)
        combination = add_ranks(combination)
        all_predictions.append(combination)
        aps.append(find_ap(combination))
    return sum(aps) / len(aps), aps, pd.concat(all_predictions)

mean_ap, aps, all_predictions = backtest(stats, reg, years[28:], predictors)
mean_ap

mean_ap, aps, all_predictions = backtest(stats, rf, years[28:], predictors)
mean_ap