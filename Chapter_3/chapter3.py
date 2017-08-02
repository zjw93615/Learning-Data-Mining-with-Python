import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

data_filename = "basketball.csv"
results = pd.read_csv(data_filename, parse_dates=["Date"])

results.columns = ["Date", "Time", "Visitor Team", "VisitorPts", "Home Team", "HomePts", "Score Type", "OT?", "Notes"]

results["HomeWin"] = results["VisitorPts"] < results["HomePts"]
# Our "class values"
y_true = results["HomeWin"].values
results["HomeLastWin"] = False
results["VisitorLastWin"] = False

won_last = defaultdict(int)

for index, row in results.iterrows():  # Note that this is not efficient
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    results.ix[index] = row
    # Set current win
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]

X_previouswins = results[["HomeLastWin", "VisitorLastWin"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_previouswins, y_true, scoring='accuracy')
print("Using just the last result from the home and visitor teams")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

ladder = pd.read_csv("leagues_NBA_2013_standings_expanded-standings.csv", skiprows=1)

# We can create a new feature -- HomeTeamRanksHigher\
results["HomeTeamRanksHigher"] = 0
for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"
    if home_team == "Charlotte Hornets":
        home_team = "Charlotte Bobcats"
    elif visitor_team == "Charlotte Hornets":
        visitor_team = "Charlotte Bobcats"
    home_rank = ladder[ladder["Team"] == home_team]["Rk"].values[0]
    visitor_rank = ladder[ladder["Team"] == visitor_team]["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
    results.ix[index] = row

X_homehigher =  results[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring='accuracy')
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

from sklearn.grid_search import GridSearchCV

parameter_space = {
                   "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                   }
clf = DecisionTreeClassifier(random_state=14)
grid = GridSearchCV(clf, parameter_space)
grid.fit(X_homehigher, y_true)
print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
