# NBA-MVP
-- Work in progress
-- This project is carried out with the steps indicated in videos on the YouTube channel "Dataquest"

This is a project where we will try to predict which NBA player will win the MVP.

The information is obtained from the web "https://www.basketball-reference.com/".

We will download the files using requests and selenium, then parse them with beautifulsoup and load them into pandas DataFrames.

We will combine our mvp, player and team stats data using pandas. During data cleaning we will work with the merge, assign, fill and replace functions. We will also use matplotlib to explore the data.

We'll first prepare the data for machine learning and use a Ridge Regression model. We'll then define an error metric, backtest across most of the data set, and iterate on our predictors. We'll end by using a Random Forest model to make predictions.

