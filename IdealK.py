import numpy as np
import pandas as pd
from datetime import datetime
from math import inf
from sklearn.metrics import mean_squared_error

start = datetime.now()

pd.set_option('display.max_rows', None)
PATH = "C:/Users/aw/Documents/WashU/2022-2023/Spring Semester/Math 401/OE Match Data/" # Edit this variable accordingly

def get_player_data(df, playername):
    # Given a DataFrame, finds all rows corresponding to that player. Takes into account the possibility of name changes.

    playerid = list(set(df[(df["playername"]==playername)]["playerid"]))
    assert len(playerid) == 1

    return df[(df["playerid"]==playerid[0])]

def get_players(df):
    # Given a DataFrame, returns a list of all unique player names.

    return list(set(df["playername"]))

def new_elo(row, k, mode=1):
    # Given a row corresponding to a player, calculates the new ELO of the player and the opponent based on the gold statistics.
    # Mode 1: default, accounts for magnitude of gold difference
    # Mode 2: does not account for magnitude of gold difference, result is 1, 0.5, or 0 depending on gold difference

    assert mode in [1, 2]
    elo = row["ELO"]
    opp_elo = row["opp_ELO"]
    constant = norm_constants[row["position"]]
    variance = 400

    match mode:
        case 1:
            actual = constant*row["golddiff"] # actual performance
            expected = (elo-opp_elo) # expected performance
            elo += k*f(actual-expected, variance)
            opp_elo += k*f(expected-actual, variance)
        case 2:
            golddiff = row["golddiff"]
            n = 500
            if golddiff > n:
                actual = 1 # "win"
                multiplier = (golddiff/n + 3)**0.8/(7.5 + 0.006*(elo-opp_elo))
            if abs(golddiff) <= n:
                actual = 0.5 # "draw"
                multiplier = 1
            if golddiff < -n:
                actual = 0 # "lose"
                multiplier = (-golddiff/n + 3)**0.8/(7.5 + 0.006*(opp_elo-elo))
            expected = g(elo-opp_elo, variance)
            elo += k*multiplier*(actual-expected)
            opp_elo += k*multiplier*(expected-actual)

    return (elo, opp_elo)

def f(x, variance):
    # Sigmoid function, used in updating ELO.

    return 1/(1+10**(-x/variance))-0.5

def g(x, variance):
    # Alternate sigmoid function

    return 1/(1+10**(-x/variance))

ideal_share = {"top": 0.215, "jng": 0.185, "mid": 0.226, "bot": 0.241, "sup": 0.133} # Globally defined, ideal gold share by position averaged 2019 Spring-2023 Spring
ideal_gold = {"top": 11445.894, "jng": 9890.908, "mid": 12046.563, "bot": 12855.070, "sup": 7122.977} # Also globally defined, averaged 2019 Spring-2023 Spring
norm_constants = {} # 1000 ELO = eta * gold
for k, v in ideal_gold.items():
    norm_constants[k] = 1000/v

class ELO:
    # Packages a dictionary of player ELOs with the DataFrame it came from
    def __init__(self, elos, df):
        self.elos = elos
        self.df = df
    
    def __repr__(self):
        return "ELOs:%s\nDF: %s" % (self.elos, self.df)
    
    def get_player_elo(self, playername):
        # Given a player name, returns their ELO.
        return self.elos[playername]

    def get_team_elo(self, team):
        # Given a list of players, returns the sum of their ELOs.
        return sum(list(map(self.get_player_elo, team)))

def update_elos(df, k, mode=1):
    # Given a DataFrame, loops through all of the rows and updates each player's ELO recursively. Assumes the DataFrame is in chronological order.
    # Mode 1: default, accounts for magnitude of gold difference
    # Mode 2: does not account for magnitude of gold difference, result is 1, 0.5, or 0 depending on gold difference

    assert mode in [1, 2]
    df = df.copy()
    players = get_players(df)
    elos = {player:1000 for player in players} # locally stores ELO

    split = df.iloc[0]["split"]
    year = df.iloc[0]["year"]

    for index, row in df.iterrows():
        # Regress to the mean every year and split
        if row["year"] != year:
            j = 0.10

            year = row["year"]
            split = row["split"]
            for i, v in elos.items():
                elos[i] = 1000*j + v*(1-j)
   
        if row["split"] != split:
            j = 0.05

            split = row["split"]

            for i, v in elos.items():
                elos[i] = 1000*j + v*(1-j)

        # Update ELOs for each row but only calculate new ELO on red side, because of nearly duplicate columns
        if row["side"] == "Blue":
            df.at[index, "ELO"] = elos[row["playername"]] 
            df.at[index, "opp_ELO"] = elos[row["opp_playername"]]
        
        if row["side"] == "Red":
            df.at[index, "ELO"] = elos[row["playername"]] 
            df.at[index, "opp_ELO"] = elos[row["opp_playername"]]

            match mode:
                case 1:
                    new = new_elo(df.loc[index], k)
                    elos[row["playername"]] = new[0]
                    elos[row["opp_playername"]] = new[1]
                case 2:
                    new = new_elo(df.loc[index], k, mode=2)
                    elos[row["playername"]] = new[0]
                    elos[row["opp_playername"]] = new[1]
    
    sorted_elos = sorted(elos.items(), key=lambda x:x[1])
    return ELO(dict(sorted_elos), df)

def add_actual(row):
    # Given a row, adds the actual result.
    n = 500
    golddiff = row["golddiff"]
    if golddiff > n:
        row["actual"] = 1 # "win"
    if abs(golddiff) <= n:
        row["actual"] = 0.5 # "draw"
    if golddiff < -n:
        row["actual"] = 0 # "lose"
    return row

def add_constants(row):
    # Given a row, adds the normalizing constant.
    row["constant"] = norm_constants[row["position"]]
    return row

def idealK(df, mode=1):
    # Given a DataFrame, finds the ideal K-factor for updating ELO by minimizing the Mean Squared Error
    arr = np.arange(20, 50, 0.5)
    ideal = 0
    min_mse = inf
    variance = 400
    df = df.apply(lambda x: add_constants(x), axis=1)
    df = df.apply(lambda x: add_actual(x), axis=1)
    
    for k in arr:
        print("Evaluating with K = ", k)

        new_df = update_elos(df, k, mode=mode).df
        playoffs = new_df[(new_df["playoffs"]==1)]
        
        match mode:
            case 1:
                mse = mean_squared_error(new_df["constant"]*new_df["golddiff"], new_df["ELO"]-new_df["opp_ELO"])
            case 2:
                mse = mean_squared_error(new_df["actual"], g(new_df["ELO"]-new_df["opp_ELO"], variance))
        print("MSE: ", mse)
        if mse < min_mse:
            ideal = k
        min_mse = min(min_mse, mse)
    return ideal

# --- DATA COLLECTION ---
LCS = pd.read_csv(PATH + "LCS.csv")
LEC = pd.read_csv(PATH + "LEC.csv")
LCK = pd.read_csv(PATH + "LCK.csv")
LPL = pd.read_csv(PATH + "LPL.csv")

print(idealK(LCS))
print(idealK(LCS, mode=2))
print(idealK(LEC))
print(idealK(LEC, mode=2))
print(idealK(LCK))
print(idealK(LCK, mode=2))
print(idealK(LPL))
print(idealK(LPL, mode=2))

end = datetime.now()
print("Runtime: ", end-start)