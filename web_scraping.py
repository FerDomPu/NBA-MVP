# Import
import requests
import os
from bs4 import BeautifulSoup
import pandas as pd
import time
from selenium import webdriver
from chromedriver_py import binary_path # this will get you the path variable

# URL format with MVP data
url_format = 'https://www.basketball-reference.com/awards/awards_{}.html'

# Years in which we will obtain information
years = list(range(1990,2023))

# Create HTML files with the web information for the indicated years: MVPs
for year in years:
    url = url_format.format(year)
    data = requests.get(url)
    with open('data/mvp/{}.html'.format(year),'w', encoding="utf-8") as f:
        f.write(data.text)
        
# Create a joint table with the data of the MVPs of the different years
dfs = []
for year in years:
    with open("data/mvp/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
        # Use BeautifulSoup to read html
        soup = BeautifulSoup(page, 'html.parser')       
    # Remove over_header
    soup.find('tr', class_="over_header").decompose()
    # Find table
    mvp_table = soup.find_all(id="mvp")[0]
    # Pandas read html   
    mvp_df = pd.read_html(str(mvp_table))[0]
    mvp_df["Year"] = year
    dfs.append(mvp_df)

mvps = pd.concat(dfs)
mvps.to_csv('mvps.csv')

# Create HTML files with the web information for the indicated years: Players
player_stats_url = "https://www.basketball-reference.com/leagues/NBA_{}_per_game.html"

# # Use chromedriver
# svc = webdriver.ChromeService(executable_path=binary_path)
# driver = webdriver.Chrome(service=svc)

for year in years:
    url = player_stats_url.format(year)
    # driver.get(url)
    # driver.execute_script("window.scrollTo(1,10000)")
    # time.sleep(2)  
    # with open("data/player/{}.html".format(year), "w", encoding='utf-8') as f:
    #     f.write(driver.page_source)
    data = requests.get(url)
    if year == 2005:
        time.sleep(65)
    with open('data/player/{}.html'.format(year),'w', encoding="utf-8") as f:
        f.write(data.text)
    
# Create a joint table with the data of the player stats of the different years
dfs = []
for year in years:
    with open("data/player/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
        # Use BeautifulSoup to read html
        soup = BeautifulSoup(page, 'html.parser')       
    # Remove thead
    for thead in soup.find_all('tr',class_='thead'):
        thead.extract()
    # Find table
    player_table = soup.find(id="per_game_stats")
    # Pandas read html   
    player_df = pd.read_html(str(player_table))[0]
    player_df["Year"] = year
    dfs.append(player_df)

players = pd.concat(dfs)
players.to_csv('players.csv')

# Create HTML files with the web information for the indicated years: Standings
standings_url = "https://www.basketball-reference.com/leagues/NBA_{}_standings.html"

for year in years:
    url = standings_url.format(year)
    data = requests.get(url)
    if year == 2005:
        time.sleep(65)
    with open('data/team/{}.html'.format(year),'w', encoding="utf-8") as f:
        f.write(data.text)

# Create a joint table with the data of the player stats of the different years
dfs = []
for year in years:
    with open("data/team/{}.html".format(year), encoding="utf-8") as f:
        page = f.read()
        # Use BeautifulSoup to read html
        soup = BeautifulSoup(page, 'html.parser')      
    # Find table East
    e_conference_table = soup.find(id="divs_standings_E")
    # Remove thead
    for thead in e_conference_table.find_all('tr',class_='thead'):
        thead.decompose()
    # Pandas read html   
    e_df = pd.read_html(str(e_conference_table))[0]
    e_df.rename(columns={"Eastern Conference": "Team"},inplace=True)
    e_df["Year"] = year
    dfs.append(e_df)
    
    # Find table West
    w_conference_table = soup.find(id="divs_standings_W")
    # Remove thead
    for thead in w_conference_table.find_all('tr',class_='thead'):
        thead.decompose()
    # Pandas read html   
    w_df = pd.read_html(str(w_conference_table))[0]
    w_df.rename(columns={"Western Conference": "Team"},inplace=True)
    w_df["Year"] = year
    dfs.append(w_df)
    
teams = pd.concat(dfs)
teams.to_csv('teams.csv')