# Run this file once to get all of the necessary CSV files.

import numpy as np
import pandas as pd
from datetime import datetime

start = datetime.now()

pd.set_option("display.max_rows", None)
PATH = "C:/Users/aw/Documents/WashU/2022-2023/Spring Semester/Math 401/OE Match Data/" # Edit this variable accordingly

Data2019 = pd.read_csv(PATH + "2019_Lol_esports_match_data_from_OraclesElixir.csv")
Data2020 = pd.read_csv(PATH + "2020_Lol_esports_match_data_from_OraclesElixir.csv")
Data2021 = pd.read_csv(PATH + "2021_Lol_esports_match_data_from_OraclesElixir.csv")
Data2022 = pd.read_csv(PATH + "2022_Lol_esports_match_data_from_OraclesElixir.csv")
Data2023 = pd.read_csv(PATH + "2023_Lol_esports_match_data_from_OraclesElixir.csv")

# --- DEFINITIONS ---

def get_split_data(league, year, season):
    # Returns a DataFrame containing match data from the given league, year, and season, removing total team data.
    # Adds some columns for calculation.
    # Also populates each row with opponent data using get_opp_data() and adds team gold data using get_team_gold().

    assert league in ["LCS", "LEC", "LCK", "LPL"] # There are other leagues possible, but for our purposes we will only allow these four.
    assert 2019<=year<=2023
    assert season in ["Winter", "Spring", "Summer"]
    if season == "Winter": assert (year == 2023) & (league == "LEC")

    match year:
        case 2019:
            data = Data2019
        case 2020:
            data = Data2020
        case 2021:
            data = Data2021
        case 2022:
            data = Data2022
        case 2023:
            data = Data2023

    data = data[(data.league==league) & (data.split==season) & (data.participantid<100)] # removes team data
    data = data.apply(lambda x: normalize_data(x), axis=1) # normalizes to 30 minutes, all additional gold rows should be normalized
    data["ELO"] = 1000 # initializes all ELO, these will be updated later on

    criteria = ["gameid", "league", "year", "split", "playoffs", "date", "patch", "participantid", "side", "position", "playername", "playerid", "teamname", "gamelength", "result", "totalgold", "ELO"] # columns we are interested in
    data = data.filter(items=criteria)
    data = data.apply(lambda x: get_opp_data(data, x), axis=1) # adds opponent data
    data = data.apply(lambda x: get_gold_diff(x), axis=1) # adds gold difference

    return data

def get_opp_data(df, row):
    # Given a DataFrame and a row of the DataFrame corresponding to a player, finds the necessary data for the role opponent of the player and adds it to the row.
    # Checks that there is exactly one opponent and that the opponent has the same gameid, that exactly one of the two players won the game, and that the opposing positions are the same.
    # This accounts for the possibility of in-game lane swaps by always matching the position in the database, regardless of the player's normal position. In the future, it may be useful to omit games containing lane swaps.

    if row["side"] == "Blue":
        assert len(df[(df["gameid"]==row["gameid"]) & (df["position"]==row["position"]) & (df["side"]=="Red")])==1
        opp = df[(df["gameid"]==row["gameid"]) & (df["position"]==row["position"]) & (df["side"]=="Red")].iloc[0]
    if row["side"] == "Red":
        assert len(df[(df["gameid"]==row["gameid"]) & (df["position"]==row["position"]) & (df["side"]=="Blue")])==1
        opp = df[(df["gameid"]==row["gameid"]) & (df["position"]==row["position"]) & (df["side"]=="Blue")].iloc[0]

    assert row["result"] + opp["result"] == 1
    assert row["gameid"] == opp["gameid"]
    assert row["position"] == opp["position"]
    
    criteria = ["participantid", "side", "position", "playername", "playerid", "teamname", "result", "totalgold", "teamgold", "goldshare", "ELO", "teamELO", "ELOshare"]
    opp = opp.filter(items=criteria)
    opp = opp.add_prefix("opp_")
    row = row._append(opp)
    
    return row

def get_gold_diff(row):
    # Given a row corresponding to a player, finds the player's gold difference with his row opponent by calculating totalgold - opp_totalgold
    
    row["golddiff"] = row["totalgold"] - row["opp_totalgold"]
    return row

def ideal_gold_share(df):
    # Given a DataFrame corresponding to a split, calculates the ideal gold share by position and returns it as a dictionary. This is done by calculating each player's gold share, then averaging by position.

    result = {"top": 0.0, "jng": 0.0, "mid": 0.0, "bot": 0.0, "sup": 0.0}

    top = df[(df["position"]=="top")]
    jng = df[(df["position"]=="jng")]
    mid = df[(df["position"]=="mid")]
    bot = df[(df["position"]=="bot")]
    sup = df[(df["position"]=="sup")]

    result["top"] = np.mean(top["goldshare"])
    result["jng"] = np.mean(jng["goldshare"])
    result["mid"] = np.mean(mid["goldshare"])
    result["bot"] = np.mean(bot["goldshare"])
    result["sup"] = np.mean(sup["goldshare"])

    return result

def normalize_data(row):
    # Normalizes gold data in a row based on game length to 30 minutes.
    
    row["totalgold"] = row["totalgold"]/row["gamelength"]*1800
    return row

def get_player_data(df, playername):
    # Given a DataFrame, finds all rows corresponding to that player. Takes into account the possibility of name changes.

    playerid = list(set(df[(df["playername"]==playername)]["playerid"]))
    assert len(playerid) == 1

    return df[(df["playerid"]==playerid[0])]

def get_players(df):
    # Given a DataFrame, returns a list of all unique player names.

    return list(set(df["playername"]))

# --- DATA COLLECTION ---

print("Getting data from LCS 2023 Spring...")
LCS2023_Spring = get_split_data("LCS", 2023, "Spring")
print("Done!")
print("Getting data from LEC 2023 Spring...")
LEC2023_Spring = get_split_data("LEC", 2023, "Spring")
print("Done!")
print("Getting data from LCK 2023 Spring...")
LCK2023_Spring = get_split_data("LCK", 2023, "Spring")
print("Done!")
print("Getting data from LPL 2023 Spring...")
LPL2023_Spring = get_split_data("LPL", 2023, "Spring")
print("Done!")
print("Getting data from LEC 2023 Winter...")
LEC2023_Winter = get_split_data("LEC", 2023, "Winter")
print("Done!")

print("Getting data from LCS 2022 Summer...")
LCS2022_Summer = get_split_data("LCS", 2022, "Summer")
print("Done!")
print("Getting data from LEC 2022 Summer...")
LEC2022_Summer = get_split_data("LEC", 2022, "Summer")
print("Done!")
print("Getting data from LCK 2022 Summer...")
LCK2022_Summer = get_split_data("LCK", 2022, "Summer")
print("Done!")
print("Getting data from LPL 2022 Summer...")
LPL2022_Summer = get_split_data("LPL", 2022, "Summer")
print("Done!")

print("Getting data from LCS 2022 Spring...")
LCS2022_Spring = get_split_data("LCS", 2022, "Spring")
print("Done!")
print("Getting data from LEC 2022 Spring...")
LEC2022_Spring = get_split_data("LEC", 2022, "Spring")
print("Done!")
print("Getting data from LCK 2022 Spring...")
LCK2022_Spring = get_split_data("LCK", 2022, "Spring")
print("Done!")
print("Getting data from LPL 2022 Spring...")
LPL2022_Spring = get_split_data("LPL", 2022, "Spring")
print("Done!")

print("Getting data from LCS 2021 Summer...")
LCS2021_Summer = get_split_data("LCS", 2021, "Summer")
print("Done!")
print("Getting data from LEC 2021 Summer...")
LEC2021_Summer = get_split_data("LEC", 2021, "Summer")
print("Done!")
print("Getting data from LCK 2021 Summer...")
LCK2021_Summer = get_split_data("LCK", 2021, "Summer")
print("Done!")
print("Getting data from LPL 2021 Summer...")
LPL2021_Summer = get_split_data("LPL", 2021, "Summer")
print("Done!")

print("Getting data from LCS 2021 Spring...")
LCS2021_Spring = get_split_data("LCS", 2021, "Spring")
print("Done!")
print("Getting data from LEC 2021 Spring...")
LEC2021_Spring = get_split_data("LEC", 2021, "Spring")
print("Done!")
print("Getting data from LCK 2021 Spring...")
LCK2021_Spring = get_split_data("LCK", 2021, "Spring")
print("Done!")
# LPL2021_Spring = get_split_data("LPL", 2021, "Spring") something wrong here idk what

print("Getting data from LCS 2020 Summer...")
LCS2020_Summer = get_split_data("LCS", 2020, "Summer")
print("Done!")
print("Getting data from LEC 2020 Summer...")
LEC2020_Summer = get_split_data("LEC", 2020, "Summer")
print("Done!")
print("Getting data from LCK 2020 Summer...")
LCK2020_Summer = get_split_data("LCK", 2020, "Summer")
print("Done!")
# LPL2020_Summer = get_split_data("LPL", 2020, "Summer")

print("Getting data from LCS 2020 Spring...")
LCS2020_Spring = get_split_data("LCS", 2020, "Spring")
print("Done!")
print("Getting data from LEC 2020 Spring...")
LEC2020_Spring = get_split_data("LEC", 2020, "Spring")
print("Done!")
print("Getting data from LCK 2020 Spring...")
LCK2020_Spring = get_split_data("LCK", 2020, "Spring")
print("Done!")
# LPL2020_Spring = get_split_data("LPL", 2020, "Spring")

print("Getting data from LCS 2019 Summer...")
LCS2019_Summer = get_split_data("LCS", 2019, "Summer")
print("Done!")
print("Getting data from LEC 2019 Summer...")
LEC2019_Summer = get_split_data("LEC", 2019, "Summer")
print("Done!")
print("Getting data from LCK 2019 Summer...")
LCK2019_Summer = get_split_data("LCK", 2019, "Summer")
print("Done!")
# LPL2019_Summer = get_split_data("LPL", 2019, "Summer")

print("Getting data from LCS 2019 Spring...")
LCS2019_Spring = get_split_data("LCS", 2019, "Spring")
print("Done!")
print("Getting data from LEC 2019 Summer...")
LEC2019_Spring = get_split_data("LEC", 2019, "Spring")
print("Done!")
print("Getting data from LCK 2019 Summer...")
LCK2019_Spring = get_split_data("LCK", 2019, "Spring")
print("Done!")
# LPL2019_Spring = get_split_data("LPL", 2019, "Spring")


# --- DATA PREPARATION FOR PLAYOFF PREDICTIONS ---
# Code is inefficient right now

print("Writing LCS files...")
LCS_splits = [LCS2023_Spring, 
              LCS2022_Summer, LCS2022_Spring, 
              LCS2021_Summer, LCS2021_Spring, 
              LCS2020_Summer, LCS2020_Spring, 
              LCS2019_Summer, LCS2019_Spring]
LCS_splits.reverse()
LCS = pd.concat(LCS_splits, ignore_index=True)
LCS.to_csv(PATH + "LCS.csv")
LCS_splits_new = []
for i in range(len(LCS_splits)):
    df = pd.concat(LCS_splits[:i+1], ignore_index=True)
    LCS_splits_new.append(df)
    df.to_csv(PATH + "LCS" + str(i) + ".csv")
print("Done!")

print("Writing LEC files...")
LEC_splits = [LEC2023_Spring, LEC2023_Winter, 
              LEC2022_Summer, LEC2022_Spring, 
              LEC2021_Summer, LEC2021_Spring, 
              LEC2020_Summer, LEC2020_Spring, 
              LEC2019_Summer, LEC2019_Spring]
LEC_splits.reverse()
LEC = pd.concat(LEC_splits, ignore_index=True)
LEC.to_csv(PATH + "LEC.csv")
LEC_splits_new = []
for i in range(len(LEC_splits)):
    df = pd.concat(LEC_splits[:i+1], ignore_index=True)
    LEC_splits_new.append(df)
    df.to_csv(PATH + "LEC" + str(i) + ".csv")
print("Done!")

print("Writing LCK files...")
LCK_splits = [LCK2023_Spring, 
              LCK2022_Summer, LCK2022_Spring, 
              LCK2021_Summer, LCK2021_Spring, 
              LCK2020_Summer, LCK2020_Spring, 
              LCK2019_Summer, LCK2019_Spring]
LCK_splits.reverse()
LCK = pd.concat(LCK_splits, ignore_index=True)
LCK.to_csv(PATH + "LCK.csv")
LCK_splits_new = []
for i in range(len(LCK_splits)):
    df = pd.concat(LCK_splits[:i+1], ignore_index=True)
    LCK_splits_new.append(df)
    df.to_csv(PATH + "LCK" + str(i) + ".csv")
print("Done!")

print("Writing LPL files...")
LPL_splits = [LPL2023_Spring, 
              LPL2022_Summer, LPL2022_Spring, 
              LPL2021_Summer] # LPL2021_Spring, 
              # LPL2020_Summer, LPL2020_Spring, 
              # LPL2019_Summer, LPL2019_Spring]
LPL_splits.reverse()
LPL = pd.concat(LPL_splits, ignore_index=True)
LPL.to_csv(PATH + "LPL.csv")
LPL_splits_new = []
for i in range(len(LPL_splits)):
    df = pd.concat(LPL_splits[:i+1], ignore_index=True)
    LPL_splits_new.append(df)
    df.to_csv(PATH + "LPL" + str(i) + ".csv")
print("Done!")

end = datetime.now()
print("Runtime: ", end-start)