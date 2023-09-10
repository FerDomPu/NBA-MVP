# Import
import pandas as pd

# Create and modify MVPs DataFrame
mvps = pd.read_csv('mvps.csv')
mvps = mvps[['Player', 'Year', 'Pts Won', 'Pts Max', 'Share']]
mvps.head()

# Create and modify Players DataFrame
players = pd.read_csv('players.csv')
del players['Unnamed: 0']
del players['Rk']
players['Player'] = players['Player'].str.replace('*','', regex=False)

# Delete the repeated rows of players who have played in more than one 
# team in a season, keeping the total and the last team
def single_row(df):
    if df.shape[0] == 1:
        return df
    else:
        row = df[df['Tm'] == 'TOT']
        row['Tm'] = df.iloc[-1,:]['Tm']
        
players = players.groupby(['Player', 'Year']).apply(single_row)
players.index = players.index.droplevel([0,1])

# Combining the Player and MVP data
combined = players.merge(mvps, how='outer', on=['Player','Year'])
combined[combined.columns[-3:]] = combined[combined.columns[-3:]].fillna(0)

# Cleaning the team data
teams = pd.read_csv('teams.csv')
teams['Team'] = teams['Team'].str.replace('*','', regex=False)

# Create a dictionary with team abbreviations
nicknames = {}

with open('nicknames.txt') as f:
    lines = f.readlines()
    for line in lines[1:]:
        abbrev,name = line.replace('\n','').split(',')
        nicknames[abbrev] = name

# Create a new column with Team
combined['Team'] = combined['Tm'].map(nicknames)

# Combine combined and teams
stats = combined.merge(teams, how='outer', on=['Team','Year'])
del stats['Unnamed: 0']
stats['GB'] = stats['GB'].str.replace('—','0')
stats['GB'] = pd.to_numeric(stats['GB'])
stats = stats.apply(pd.to_numeric, errors='ignore')

# Combining the Team, Player and MVP data
stats.to_csv('player_mvp_stats.csv')

# Exploring data
import matplotlib.pyplot as plt

highest_scoring = stats[stats["G"] > 70].sort_values("PTS", ascending=False).head(10)
highest_scoring.plot.bar("Player", "PTS")

highest_scoring_by_year = stats.groupby("Year").apply(lambda x: x.sort_values("PTS", ascending=False).head(1))
highest_scoring_by_year.plot.bar("Year", "PTS")

stats.corr(numeric_only=True)["Share"]
