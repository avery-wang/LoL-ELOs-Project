import numpy as np
import pandas as pd
from datetime import datetime

start = datetime.now()

pd.set_option("display.max_rows", None)
PATH = "C:/Users/aw/Documents/WashU/2022-2023/Spring Semester/Math 401/OE Match Data/" # Edit this variable accordingly

def get_player_data(df, playername):
    # Given a DataFrame, finds all rows corresponding to that player. Takes into account the possibility of name changes.

    playerid = list(set(df[(df["playername"]==playername)]["playerid"]))
    assert len(playerid) == 1

    return df[(df["playerid"]==playerid[0])]

def get_players(df):
    # Given a DataFrame, returns a list of all unique player names.

    return list(set(df["playername"]))

def new_elo(row, mode=1):
    # Given a row corresponding to a player, calculates the new ELO of the player and the opponent based on the gold statistics.
    # Mode 1: default, modified
    # Mode 2: traditional, result is 1, 0.5, or 0 depending on gold difference
    # K-factors were calculated using IdealK.py, minimizing the Mean Squared Error. 
    # The LCS and LPL have K-factors slightly lower and the LEC and LCK have slightly higher, the K-factors chosen are around the median.

    assert mode in [1, 2]
    elo = row["ELO"]
    opp_elo = row["opp_ELO"]
    constant = norm_constants[row["position"]]
    variance = 400

    match mode:
        case 1:
            k = 25 # K-factor
            actual = constant*row["golddiff"] # actual performance
            expected = (elo-opp_elo) # expected performance
            elo += k*f(actual-expected, variance)
            opp_elo += k*f(expected-actual, variance)
        case 2:
            k = 30 # K-factor
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


def update_elos(df, mode=1):
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
            k = 0.10

            year = row["year"]
            split = row["split"]
            for i, v in elos.items():
                elos[i] = 1000*k + v*(1-k)
   
        if row["split"] != split:
            k = 0.05

            split = row["split"]

            for i, v in elos.items():
                elos[i] = 1000*k + v*(1-k)

        # Update ELOs for each row but only calculate new ELO on red side, because of nearly duplicate columns
        if row["side"] == "Blue":
            df.at[index, "ELO"] = elos[row["playername"]] 
            df.at[index, "opp_ELO"] = elos[row["opp_playername"]]
        
        if row["side"] == "Red":
            df.at[index, "ELO"] = elos[row["playername"]] 
            df.at[index, "opp_ELO"] = elos[row["opp_playername"]]

            match mode:
                case 1:
                    new = new_elo(df.loc[index])
                    elos[row["playername"]] = new[0]
                    elos[row["opp_playername"]] = new[1]
                case 2:
                    new = new_elo(df.loc[index], mode=2)
                    elos[row["playername"]] = new[0]
                    elos[row["opp_playername"]] = new[1]
    
    sorted_elos = sorted(elos.items(), key=lambda x:x[1])
    return ELO(dict(sorted_elos), df)

class Prediction:
    # Stores the result of a prediction using ELOs. Lists the winner first, so the percentage must always be at least 50%.
    def __init__(self, team, elo, opp, opp_elo, bestof, percentage):
        assert bestof in [1, 3, 5]
        assert 0.5 <= percentage <= 1
        self.team = team
        self.elo = elo
        self.opp = opp
        self.opp_elo = opp_elo
        self.bestof = bestof
        self.percentage = percentage
    
    def __repr__(self):
        return "Winner: %s\nWinner ELO: %s\nLoser: %s\nLoser ELO: %s\nBest-of-%s\nProbability: %s" % (self.team, self.elo, self.opp, self.opp_elo, self.bestof, self.percentage)

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
    
    def get_player_elo_prog(self, playername):
        # Given a player name, returns an array showing a player's ELO progression over time.
        return self.df[(self.df["playername"]==playername)].filter(items=["date", "teamname", "ELO"])

    def get_team_elo(self, team):
        # Given a list of players, returns the sum of their ELOs.
        return sum(list(map(self.get_player_elo, team)))
    
    def predict_winner(self, team1, team2, bestof=5):
        # Given two teams, returns a Prediction containing which team will win and percentage certainty, using the traditional ELO calculations.
        # If a perfect 50-50 tie is predicted, returns a Prediction with the teams in the same order as they were inputted and 50%.
        # Default is Best-of-5 because that is the most common.

        assert bestof in [1, 3, 5]
        elo1 = self.get_team_elo(team1)
        elo2 = self.get_team_elo(team2)
        variance = 400

        if elo1 > elo2:
            winner = team1
            winner_elo = elo1
            loser = team2
            loser_elo = elo2
            prob = g(elo1-elo2, variance)
        if elo1 < elo2:
            winner = team2
            winner_elo = elo2
            loser = team1
            loser_elo = elo1
            prob = g(elo2-elo1, variance)
        if elo1 == elo2:
            return Prediction(team1, elo1, team2, elo2, bestof, 0.5)
        
        match bestof:
            case 1:
                return Prediction(winner, winner_elo, loser, loser_elo, bestof, prob)
            case 3:
                prob = prob**2 + prob**2*(1-prob)*2
                return Prediction(winner, winner_elo, loser, loser_elo, bestof, prob)
            case 5:
                prob = prob**3 + prob**3*(1-prob)*3 + prob**3*(1-prob)**2*6
                return Prediction(winner, winner_elo, loser, loser_elo, bestof, prob)
    
    # --- PREDICTION FUNCTIONS ---
    # Each function assumes the teams are listed in order of seeding. This may need to be changed depending on situations where a team can choose its opponent.

    def LCS_Playoffs_Old(self, team1, team2, team3, team4, team5, team6, third_place=False):
        # Given the 6 teams (lists of 5 players) that make the LCS playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # Includes an option for a Third-place match; this is only played in Summer.
        # This is the Single Elimination version used in 2019.

        # Quarterfinal 1
        predict1 = self.predict_winner(team4, team5)
        print("Quarterfinal 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team

        # Quarterfinal 2
        predict2 = self.predict_winner(team3, team6)
        print("Quarterfinal 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team

        # Semifinal 1
        predict3 = self.predict_winner(team1, winner1)
        print("Semifinal 1")
        print(predict3)
        print("\n")
        winner3 = predict3.team
        loser3 = predict3.opp

        # Semifinal 2
        predict4 = self.predict_winner(team2, winner2)
        print("Semifinal 2")
        print(predict4)
        print("\n")
        winner4 = predict4.team
        loser4 = predict4.opp

        if third_place:
            # Third-place match
            predict5 = self.predict_winner(loser3, loser4)
            print("Third-place match")
            print(predict5)
            print("\n")
            winner5 = predict5.team

        # Final
        predict6 = self.predict_winner(winner3, winner4)
        print("Final")
        print(predict6)
        print("\n")
        winner6 = predict6.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner6)
        print("Runner-up:")
        print(predict6.opp)
        if third_place:
            print("Third place:")
            print(winner5)
        print("\n\n\n")

        if third_place:
            return np.array([winner1, winner2, winner3, winner4, winner5, winner6])
        else:
            return np.array([winner1, winner2, winner3, winner4, winner6])

    def LCSSpring_Playoffs_New(self, team1, team2, team3, team4, team5, team6):
        # Given the 6 teams (lists of 5 players) that make the LCS Spring playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the Double Elimination version used from 2020-present.

        # Winner's Semifinal 1
        predict1 = self.predict_winner(team1, team4)
        print("Winner's Semifinal 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team
        loser1 = predict1.opp

        # Winner's Semifinal 2
        predict2 = self.predict_winner(team2, team3)
        print("Winner's Semifinal 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team
        loser2 = predict2.opp

        # Loser's Semifinal 1
        predict3 = self.predict_winner(team6, loser2)
        print("Loser's Semifinal 1")
        print(predict3)
        print("\n")
        winner3 = predict3.team

        # Loser's Semifinal 2
        predict4 = self.predict_winner(team5, loser1)
        print("Loser's Semifinal 2")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Winner's Final
        predict5 = self.predict_winner(winner1, winner2)
        print("Winner's Final")
        print(predict5)
        print("\n")
        winner5 = predict5.team
        loser5 = predict5.opp

        # Loser's Final
        predict6 = self.predict_winner(winner3, winner4)
        print("Loser's Final")
        print(predict6)
        print("\n")
        winner6 = predict6.team

        # Semifinal
        predict7 = self.predict_winner(loser5, winner6)
        print("Semifinal")
        print(predict7)
        print("\n")
        winner7 = predict7.team

        # Final
        predict8 = self.predict_winner(winner5, winner7)
        print("Final")
        print(predict8)
        print("\n")
        winner8 = predict8.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner8)
        print("Runner-up:")
        print(predict8.opp)
        print("Third place:")
        print(predict7.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4, winner5, winner6, winner7, winner8])

    def LCSSummer_Playoffs_New(self, team1, team2, team3, team4, team5, team6, team7, team8):
        # Given the 8 teams (lists of 5 players) that make the LCS Summer playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the Double Elimination version used from 2020-present.

        # Winner's Quarterfinal 1
        predict1 = self.predict_winner(team3, team6)
        print("Winner's Quarterfinal 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team
        loser1 = predict1.opp

        # Winner's Quarterfinal 2
        predict2 = self.predict_winner(team4, team5)
        print("Winner's Quarterfinal 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team
        loser2 = predict2.opp

        # Loser's Quarterfinal 1
        predict3 = self.predict_winner(team8, loser2)
        print("Loser's Quarterfinal 2")
        print(predict3)
        print("\n")
        winner3 = predict3.team

        # Loser's Quarterfinal 2
        predict4 = self.predict_winner(team7, loser1)
        print("Loser's Quarterfinal 2")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Winner's Semifinal 1
        predict5 = self.predict_winner(team1, winner2)
        print("Winner's Semifinal 1")
        print(predict5)
        print("\n")
        winner5 = predict5.team
        loser5 = predict5.opp

        # Winner's Semifinal 2
        predict6 = self.predict_winner(team2, winner1)
        print("Winner's Semifinal 2")
        print(predict6)
        print("\n")
        winner6 = predict6.team
        loser6 = predict6.opp

        # Loser's Semifinal 1
        predict7 = self.predict_winner(winner3, loser5)
        print("Loser's Semifinal 1")
        print(predict7)
        print("\n")
        winner7 = predict7.team

        # Loser's Semifinal 2
        predict8 = self.predict_winner(winner4, loser6)
        print("Loser's Semifinal 2")
        print(predict8)
        print("\n")
        winner8 = predict8.team

        # Winner's Final
        predict9 = self.predict_winner(winner5, winner6)
        print("Winner's Final")
        print(predict9)
        print("\n")
        winner9 = predict9.team
        loser9 = predict9.opp

        # Loser's Final
        predict10 = self.predict_winner(winner7, winner8)
        print("Loser's Final")
        print(predict10)
        print("\n")
        winner10 = predict10.team

        # Semifinal
        predict11 = self.predict_winner(loser9, winner10)
        print("Semifinal")
        print(predict11)
        print("\n")
        winner11 = predict11.team

        # Final
        predict12 = self.predict_winner(winner9, winner11)
        print("Final")
        print(predict12)
        print("\n")
        winner12 = predict12.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner12)
        print("Runner-up:")
        print(predict12.opp)
        print("Third place:")
        print(predict11.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4, winner5, winner6, winner7, winner8, winner9, winner10, winner11, winner12])
    
    def LEC_Playoffs_Old(self, team1, team2, team3, team4, team5, team6):
        # Given the 6 teams (lists of 5 players) that make the LEC playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the format used in 2019.

        # Loser's Semifinal 1
        predict1 = self.predict_winner(team3, team6)
        print("Loser's Semifinal 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team

        # Loser's Semifinal 2
        predict2 = self.predict_winner(team4, team5)
        print("Loser's Semifinal 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team

        # Winner's Final
        predict3 = self.predict_winner(team1, team2)
        print("Winner's Final")
        print(predict3)
        print("\n")
        winner3 = predict3.team
        loser3 = predict3.opp

        # Loser's Final
        predict4 = self.predict_winner(winner1, winner2)
        print("Loser's Final")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Semifinal
        predict5 = self.predict_winner(loser3, winner4)
        print("Semifinal")
        print(predict5)
        print("\n")
        winner5 = predict5.team

        # Final
        predict6 = self.predict_winner(winner3, winner5)
        print("Final")
        print(predict6)
        print("\n")
        winner6 = predict6.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner6)
        print("Runner-up:")
        print(predict6.opp)
        print("Third place:")
        print(predict5.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4, winner5, winner6])
    
    def LEC_Playoffs_Gauntlet(self, team1, team2, team3, team4, team5, team6, choice=True):
        # Given the 6 teams (lists of 5 players) that make the LEC playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # The choice variable is True if team1 chooses to play team4 in the first round and False if team1 chooses to play team3 in the first round.
        # This is the format used in 2020-2022.
        
        seeds = {tuple(team1):1, tuple(team2):2, tuple(team3):3, tuple(team4):4}
        if choice:
            opp1 = team4
            opp2 = team3
        else:
            opp1 = team3
            opp2 = team4

        # Winner's Semifinal 1
        predict1 = self.predict_winner(team1, opp1)
        print("Winner's Semifinal 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team
        loser1 = predict1.opp

        # Winner's Semifinal 2
        predict2 = self.predict_winner(team2, opp2)
        print("Winner's Semifinal 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team
        loser2 = predict2.opp

        # Loser's Round 1
        predict3 = self.predict_winner(team5, team6)
        print("Loser's Round 1")
        print(predict3)
        print("\n")
        winner3 = predict3.team

        if seeds[tuple(loser1)] > seeds[tuple(loser2)]:
            opp3 = loser1
            opp4 = loser2
        else:
            opp3 = loser2
            opp4 = loser1
        
        # Loser's Round 2
        predict4 = self.predict_winner(winner3, opp3)
        print("Loser's Round 2")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Loser's Round 3
        predict5 = self.predict_winner(winner4, opp4)
        print("Loser's Round 3")
        print(predict5)
        print("\n")
        winner5 = predict5.team

        # Winner's Final
        predict6 = self.predict_winner(winner1, winner2)
        print("Winner's Final")
        print(predict6)
        print("\n")
        winner6 = predict6.team
        loser6 = predict6.opp

        # Semifinal
        predict7 = self.predict_winner(winner5, loser6)
        print("Semifinal")
        print(predict7)
        print("\n")
        winner7 = predict7.team

        # Final
        predict8 = self.predict_winner(winner6, winner7)
        print("Final")
        print(predict8)
        print("\n")
        winner8 = predict8.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner8)
        print("Runner-up:")
        print(predict8.opp)
        print("Third place:")
        print(predict7.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4, winner5, winner6, winner7, winner8])
    
    def LEC_Playoffs_New(self, team1, team2, team3, team4):
        # Given the 4 teams (lists of 5 players) that make the LEC playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the format used in 2023.
        # Note that the 2023 format has Regular Season (10 teams), Group Stage (8 teams), and Playoffs (4 teams), but only the Playoffs are marked as playoffs=1 in the CSV file.
        # This means that the 4 teams that make the playoffs are not necessarily the top 4 in Regular Season. However, Group Stage games will be used to update their ELO.

        # Winner's Final
        predict1 = self.predict_winner(team1, team2)
        print("Winner's Final")
        print(predict1)
        print("\n")
        winner1 = predict1.team
        loser1 = predict1.opp

        # Loser's Final
        predict2 = self.predict_winner(team3, team4)
        print("Loser's Final")
        print(predict2)
        print("\n")
        winner2 = predict2.team

        # Semifinal
        predict3 = self.predict_winner(loser1, winner2)
        print("Semifinal")
        print(predict3)
        print("\n")
        winner3 = predict3.team

        # Final
        predict4 = self.predict_winner(winner1, winner3)
        print("Final")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner4)
        print("Runner-up:")
        print(predict4.opp)
        print("Third place:")
        print(predict3.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4])

    def LCK_Playoffs_Old(self, team1, team2, team3, team4, team5):
        # Given the 6 teams (lists of 5 players) that make the LCK playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the format used in 2019-2020.

        # Round 1
        predict1 = self.predict_winner(team4, team5, 3)
        print("Round 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team

        # Round 2
        predict2 = self.predict_winner(team3, winner1)
        print("Round 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team

        # Round 3
        predict3 = self.predict_winner(team2, winner2)
        print("Round 3")
        print(predict3)
        print("\n")
        winner3 = predict3.team

        # Final
        predict4 = self.predict_winner(team1, winner3)
        print("Final")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner4)
        print("Runner-up:")
        print(predict4.opp)
        print("Third place:")
        print(predict3.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4])

    def LCK_Playoffs_New(self, team1, team2, team3, team4, team5, team6):
        # Given the 6 teams (lists of 5 players) that make the LCK playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the format used in 2021-2022. It is identical to the LCS playoff format in 2019.

        return self.LCS_Playoffs_Old(team1, team2, team3, team4, team5, team6)

    def LCK_Playoffs_Newer(self, team1, team2, team3, team4, team5, team6):
        # Given the 6 teams (lists of 5 players) that make the LCK playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the format used in 2023. It is identical to the LEC playoff format in 2019.

        return self.LEC_Playoffs_Old(team1, team2, team3, team4, team5, team6)

    def LPL_Playoffs(self, team1, team2, team3, team4, team5, team6, team7, team8, team9, team10):
        # Given the 10 teams (lists of 5 players) that make the LPL playoffs, prints the Prediction for each round. Returns a list of the winners of each match in order.
        # This is the format used in 2021-present.

        # Round 1 Match 1
        predict1 = self.predict_winner(team8, team9)
        print("Round 1 Match 1")
        print(predict1)
        print("\n")
        winner1 = predict1.team

        # Round 1 Match 2
        predict2 = self.predict_winner(team7, team10)
        print("Round 1 Match 2")
        print(predict2)
        print("\n")
        winner2 = predict2.team

        # Round 2 Match 1
        predict3 = self.predict_winner(team5, winner1)
        print("Round 2 Match 1")
        print(predict3)
        print("\n")
        winner3 = predict3.team

        # Round 2 Match 2
        predict4 = self.predict_winner(team6, winner2)
        print("Round 2 Match 2")
        print(predict4)
        print("\n")
        winner4 = predict4.team

        # Round 3 Match 1
        predict5 = self.predict_winner(team4, winner3)
        print("Round 3 Match 1")
        print(predict5)
        print("\n")
        winner5 = predict5.team

        # Round 3 Match 2
        predict6 = self.predict_winner(team3, winner4)
        print("Round 3 Match 2")
        print(predict6)
        print("\n")
        winner6 = predict6.team

        # Winner's Semifinal 1
        predict7 = self.predict_winner(team1, winner5)
        print("Winner's Semifinal 1")
        print(predict7)
        print("\n")
        winner7 = predict7.team
        loser7 = predict7.opp

        # Winner's Semifinal 2
        predict8 = self.predict_winner(team2, winner6)
        print("Winner's Semifinal 2")
        print(predict8)
        print("\n")
        winner8 = predict8.team
        loser8 = predict8.opp

        # Loser's Final
        predict9 = self.predict_winner(loser7, loser8)
        print("Loser's Final")
        print(predict9)
        print("\n")
        winner9 = predict9.team

        # Winner's Final
        predict10 = self.predict_winner(winner7, winner8)
        print("Winner's Final")
        print(predict10)
        print("\n")
        winner10 = predict10.team
        loser10 = predict10.opp

        # Semifinal
        predict11 = self.predict_winner(winner9, loser10)
        print("Semifinal")
        print(predict11)
        print("\n")
        winner11 = predict11.team

        # Final
        predict12 = self.predict_winner(winner10, winner11)
        print("Final")
        print(predict12)
        print("\n")
        winner12 = predict12.team
        # Print the predicted champion, runner-up, and third place team
        print("Champion:")
        print(winner12)
        print("Runner-up:")
        print(predict12.opp)
        print("Third place:")
        print(predict11.opp)
        print("\n\n\n")

        return np.array([winner1, winner2, winner3, winner4, winner5, winner6, winner7, winner8, winner9, winner10, winner11, winner12])

# --- FUNCTIONS TO SCORE BRACKETS ---
# March-Madness style. Each function takes in a NumPy Array of match winners from the ELO class's predictions, then compares it with the actual bracket and scores the bracket. Prints the raw score, the highest possible score, and returns their quotient.

def LCS_Playoffs_Old_Score(list1, list2, third_place=False):
    if third_place:
        x = 6
    else:
        x = 5
    assert len(list1) == len(list2) == x

    if third_place:
        weights = np.array([10, 10, 20, 20, 40, 40])
    else:
        weights = np.array([10, 10, 20, 20, 40])
    
    compare = list1 == list2
    acc = np.zeros(x)
    for i in range(x):
        acc[i] = all(compare[i]) # We assume that the list will be all True or all False, this will return True if all True and False if at least one False.

    score = int(sum(weights*acc))
    total = sum(weights)
    print("Raw Score: ", score)
    print("Full Score: ", total)
    return score/total

# --- DATA PREPARATION FOR PLAYOFF PREDICTIONS ---
# Code is inefficient right now

LCS_splits = []
for i in range(9):
    LCS_splits.append(pd.read_csv(PATH + "LCS" + str(i) + ".csv"))

LCS_ELOs = []
year = 2019
season = "Spring"
for i in LCS_splits:
    i = i[(i["playoffs"]!=1) | (i["year"]!=year) | (i["split"]!=season)]
    print("Updating LCS ELOs through " + str(year) + " " + season + " using modified algorithm...")
    LCS_ELOs.append(update_elos(i, mode=1))
    print("Done!")
    print("Updating LCS ELOs through " + str(year) + " " + season + " using traditional algorithm...")
    LCS_ELOs.append(update_elos(i, mode=2))
    print("Done!")
    if season == "Spring":
        season = "Summer"
        continue
    if season == "Summer":
        season = "Spring"
        year += 1

LEC_splits = []
for i in range(10):
    LEC_splits.append(pd.read_csv(PATH + "LEC" + str(i) + ".csv"))

LEC_ELOs = []
year = 2019
season = "Spring"
for i in LEC_splits:
    i = i[(i["playoffs"]!=1) | (i["year"]!=year) | (i["split"]!=season)]
    print("Updating LEC ELOs through " + str(year) + " " + season + " using modified algorithm...")
    LEC_ELOs.append(update_elos(i, mode=1))
    print("Done!")
    print("Updating LEC ELOs through " + str(year) + " " + season + " using traditional algorithm...")
    LEC_ELOs.append(update_elos(i, mode=2))
    print("Done!")
    if season == "Winter":
        season = "Spring"
        continue
    if season == "Spring":
        season = "Summer"
        continue
    if season == "Summer":
        if year == 2022:
            season = "Winter"
        else: 
            season = "Spring"
        year += 1

LCK_splits = []
for i in range(9):
    LCK_splits.append(pd.read_csv(PATH + "LCK" + str(i) + ".csv"))

LCK_ELOs = []
year = 2019
season = "Spring"
for i in LCK_splits:
    i = i[(i["playoffs"]!=1) | (i["year"]!=year) | (i["split"]!=season)]
    print("Updating LCK ELOs through " + str(year) + " " + season + " using modified algorithm...")
    LCK_ELOs.append(update_elos(i, mode=1))
    print("Done!")
    print("Updating LCK ELOs through " + str(year) + " " + season + " using traditional algorithm...")
    LCK_ELOs.append(update_elos(i, mode=2))
    print("Done!")
    if season == "Spring":
        season = "Summer"
        continue
    if season == "Summer":
        season = "Spring"
        year += 1

LPL_splits = []
for i in range(4):
    LPL_splits.append(pd.read_csv(PATH + "LPL" + str(i) + ".csv"))

LPL_ELOs = []
year = 2021
season = "Summer"
for i in LPL_splits:
    i = i[(i["playoffs"]!=1) | (i["year"]!=year) | (i["split"]!=season)]
    print("Updating LPL ELOs through " + str(year) + " " + season + " using modified algorithm...")
    LPL_ELOs.append(update_elos(i, mode=1))
    print("Done!")
    print("Updating LPL ELOs through " + str(year) + " " + season + " using traditional algorithm...")
    LPL_ELOs.append(update_elos(i, mode=2))
    print("Done!")
    if season == "Spring":
        season = "Summer"
        continue
    if season == "Summer":
        season = "Spring"
        year += 1

# --- PLAYOFF PREDICTIONS ---
# Note: all teams do not include subs, this is because the teams must be 5 players for the team ELO to be calculated properly.
# To account for this, the player that is listed on the playoff roster, if applicable, is listed. Otherwise, the player that played the majority of the games is listed, with recency used as a tiebreaker.
# Where possible, if multiple players played during playoffs, different teams are listed with each roster combination.
# In situations where teams choose their opponent, we assume that the teams would have chosen the lowest seeded opponent, unless they acted differently in real life.

print("\nLCS 2019 Spring")
LCS2019_Spring_Teams = {
    "100": ["Ssumday", "AnDa", "huhi", "Bang", "aphromoo"],
    "C9": ["Licorice", "Svenskeren", "Nisqy", "Sneaky", "Zeyzal"],
    "CG": ["Huni", "Lira", "Damonte", "Cody Sun", "Vulcan"],
    "CLG": ["Darshan", "Griffin", "PowerOfEvil", "Stixxay", "Biofrost"],
    "FOX": ["Solo", "Rush", "Fenix", "Apollo", "Hakuho"],
    "FLY": ["V1per", "Santorin", "Pobelter", "WildTurtle", "JayJ"],
    "GG": ["Hauntzer", "Contractz", "Froggen", "Deftly", "Olleh"],
    "OPT": ["Dhokla", "Meteos", "Crown", "Arrow", "Big"],
    "TL": ["Impact", "Xmithie", "Jensen", "Doublelift", "CoreJJ"],
    "TSM": ["BrokenBlade", "Akaadian", "Bjergsen", "Zven", "Smoothie"]
}

for k, v in LCS2019_Spring_Teams.items():
    print(k)
    print(LCS_ELOs[0].get_team_elo(v))
    print(LCS_ELOs[1].get_team_elo(v))

predict0 = LCS_ELOs[0].LCS_Playoffs_Old(LCS2019_Spring_Teams["TL"],
                                        LCS2019_Spring_Teams["C9"],
                                        LCS2019_Spring_Teams["TSM"],
                                        LCS2019_Spring_Teams["FLY"],
                                        LCS2019_Spring_Teams["GG"],
                                        LCS2019_Spring_Teams["FOX"])
predict1 = LCS_ELOs[1].LCS_Playoffs_Old(LCS2019_Spring_Teams["TL"],
                                        LCS2019_Spring_Teams["C9"],
                                        LCS2019_Spring_Teams["TSM"],
                                        LCS2019_Spring_Teams["FLY"],
                                        LCS2019_Spring_Teams["GG"],
                                        LCS2019_Spring_Teams["FOX"])

print(predict0)
print(predict1)

LCS2019_Spring_Actual = np.array([LCS2019_Spring_Teams["FLY"],
                                  LCS2019_Spring_Teams["TSM"],
                                  LCS2019_Spring_Teams["TL"],
                                  LCS2019_Spring_Teams["TSM"],
                                  LCS2019_Spring_Teams["TL"]])

print(LCS_Playoffs_Old_Score(predict0, LCS2019_Spring_Actual))
print(LCS_Playoffs_Old_Score(predict1, LCS2019_Spring_Actual))

print("\nLCS 2019 Summer")
LCS2019_Summer_Teams = {
    "100": ["FakeGod", "Amazing", "Ryu", "Bang", "aphromoo"],
    "C9": ["Licorice", "Svenskeren", "Nisqy", "Sneaky", "Zeyzal"],
    "CG": ["Huni", "Lira", "Damonte", "Cody Sun", "Vulcan"],
    "CLG": ["Ruin", "Griffin", "PowerOfEvil", "Stixxay", "Biofrost"],
    "FOX": ["Solo", "MikeYeung", "Fenix", "Apollo", "Hakuho"],
    "FLY": ["V1per", "Santorin", "Pobelter", "WildTurtle", "Wadid"],
    "GG": ["Hauntzer", "Contractz", "Froggen", "FBI", "huhi"],
    "OPT": ["Dhokla", "Meteos", "Crown", "Arrow", "Big"],
    "TL": ["Impact", "Xmithie", "Jensen", "Doublelift", "CoreJJ"],
    "TSM": ["BrokenBlade", "Spica", "Bjergsen", "Zven", "Smoothie"]
}

for k, v in LCS2019_Summer_Teams.items():
    print(k)
    print(LCS_ELOs[2].get_team_elo(v))
    print(LCS_ELOs[3].get_team_elo(v))

predict2 = LCS_ELOs[2].LCS_Playoffs_Old(LCS2019_Summer_Teams["TL"],
                                        LCS2019_Summer_Teams["C9"],
                                        LCS2019_Summer_Teams["CLG"],
                                        LCS2019_Summer_Teams["TSM"],
                                        LCS2019_Summer_Teams["CG"],
                                        LCS2019_Summer_Teams["OPT"],
                                        True)
predict3 = LCS_ELOs[3].LCS_Playoffs_Old(LCS2019_Summer_Teams["TL"],
                                        LCS2019_Summer_Teams["C9"],
                                        LCS2019_Summer_Teams["CLG"],
                                        LCS2019_Summer_Teams["TSM"],
                                        LCS2019_Summer_Teams["CG"],
                                        LCS2019_Summer_Teams["OPT"],
                                        True)

print(predict2)
print(predict3)

LCS2019_Summer_Actual = np.array([LCS2019_Summer_Teams["CG"],
                                  LCS2019_Summer_Teams["CLG"],
                                  LCS2019_Summer_Teams["TL"],
                                  LCS2019_Summer_Teams["C9"],
                                  LCS2019_Summer_Teams["CLG"],
                                  LCS2019_Summer_Teams["TL"]])

print(LCS_Playoffs_Old_Score(predict2, LCS2019_Summer_Actual, True))
print(LCS_Playoffs_Old_Score(predict3, LCS2019_Summer_Actual, True))

print("\nLCS 2020 Spring")
LCS2020_Spring_Teams = {
    "100": ["Ssumday", "Meteos", "ry0ma", "Cody Sun", "Stunt"],
    "C9": ["Licorice", "Blaber", "Nisqy", "Sneaky", "Zeyzal"],
    "CLG": ["Ruin", "Griffin", "Pobelter", "Stixxay", "Smoothie"],
    "DIG": ["Huni", "Armao", "Froggen", "Johnsun", "aphromoo"],
    "EG": ["Kumo", "Svenskeren", "Jiizuke", "Bang", "Zeyzal"],
    "FLY": ["V1per", "Santorin", "PowerOfEvil", "WildTurtle", "IgNar"],
    "GG": ["Hauntzer", "Closer", "Goldenglue", "FBI", "huhi"],
    "IMT": ["sOAZ", "Xmithie", "Eika", "Apollo", "Hakuho"],
    "TL": ["Impact", "Broxah", "Jensen", "Doublelift", "CoreJJ"],
    "TSM": ["BrokenBlade", "Dardoch", "Bjergsen", "Kobbe", "Biofrost"]
}

for k, v in LCS2020_Spring_Teams.items():
    print(k)
    print(LCS_ELOs[4].get_team_elo(v))
    print(LCS_ELOs[5].get_team_elo(v))

print(
LCS_ELOs[4].LCSSpring_Playoffs_New(LCS2020_Spring_Teams["C9"],
                                   LCS2020_Spring_Teams["EG"],
                                   LCS2020_Spring_Teams["FLY"],
                                   LCS2020_Spring_Teams["100"],
                                   LCS2020_Spring_Teams["TSM"],
                                   LCS2020_Spring_Teams["GG"]))
print(
LCS_ELOs[5].LCSSpring_Playoffs_New(LCS2020_Spring_Teams["C9"],
                                   LCS2020_Spring_Teams["EG"],
                                   LCS2020_Spring_Teams["FLY"],
                                   LCS2020_Spring_Teams["100"],
                                   LCS2020_Spring_Teams["TSM"],
                                   LCS2020_Spring_Teams["GG"]))

print("\nLCS 2020 Summer")
LCS2020_Summer_Teams = {
    "100": ["Ssumday", "Contractz", "ry0ma", "Cody Sun", "Poome"],
    "C9": ["Licorice", "Blaber", "Nisqy", "Sneaky", "Zeyzal"],
    "CLG": ["Ruin", "Griffin", "Pobelter", "Stixxay", "Smoothie"],
    "DIG": ["V1per", "Dardoch", "Fenix", "Johnsun", "aphromoo"],
    "EG": ["Huni", "Svenskeren", "Goldenglue", "Bang", "Zeyzal"],
    "FLY": ["Solo", "Santorin", "PowerOfEvil", "WildTurtle", "IgNar"],
    "GG": ["Hauntzer", "Closer", "Damonte", "FBI", "huhi"],
    "IMT": ["Allorim", "Xmithie", "Insanity", "Apollo", "Hakuho"],
    "TL": ["Impact", "Broxah", "Jensen", "Tactical", "CoreJJ"],
    "TSM": ["BrokenBlade", "Spica", "Bjergsen", "Doublelift", "Biofrost"] # Treatz played two series during playoffs, but Biofrost played the majority of the time
}

for k, v in LCS2020_Summer_Teams.items():
    print(k)
    print(LCS_ELOs[6].get_team_elo(v))
    print(LCS_ELOs[7].get_team_elo(v))

print(
LCS_ELOs[6].LCSSummer_Playoffs_New(LCS2020_Summer_Teams["TL"],
                                   LCS2020_Summer_Teams["C9"],
                                   LCS2020_Summer_Teams["FLY"],
                                   LCS2020_Summer_Teams["TSM"],
                                   LCS2020_Summer_Teams["GG"],
                                   LCS2020_Summer_Teams["EG"],
                                   LCS2020_Summer_Teams["100"],
                                   LCS2020_Summer_Teams["DIG"]))
print(
LCS_ELOs[7].LCSSummer_Playoffs_New(LCS2020_Summer_Teams["TL"],
                                   LCS2020_Summer_Teams["C9"],
                                   LCS2020_Summer_Teams["FLY"],
                                   LCS2020_Summer_Teams["TSM"],
                                   LCS2020_Summer_Teams["GG"],
                                   LCS2020_Summer_Teams["EG"],
                                   LCS2020_Summer_Teams["100"],
                                   LCS2020_Summer_Teams["DIG"]))

print("\nLCS 2021 Spring")
LCS2021_Spring_Teams = {
    "100": ["Ssumday", "Closer", "ry0ma", "FBI", "huhi"],
    "C9": ["Fudge", "Blaber", "Perkz", "Zven", "Vulcan"],
    "CLG": ["Finn", "Broxah", "Pobelter", "WildTurtle", "Smoothie"],
    "DIG": ["FakeGod", "Dardoch", "Soligo", "Neo", "aphromoo"],
    "EG": ["Impact", "Svenskeren", "Jiizuke", "Deftly", "IgNar"],
    "FLY": ["Licorice", "Josedeodo", "Palafox", "Johnsun", "Diamond"],
    "GG": ["Niles", "wallflower", "Ablazeolive", "Stixxay", "Newbie"],
    "IMT": ["Revenge", "Xerxe", "Insanity", "Raes", "Destiny"],
    "TL": ["Alphari", "Armao", "Jensen", "Tactical", "CoreJJ"], # Santorin played two series and Armao played two series, Armao played in the Semifinal and Final
    "TSM": ["Huni", "Spica", "PowerOfEvil", "Lost", "SwordArt"],
}

for k, v in LCS2021_Spring_Teams.items():
    print(k)
    print(LCS_ELOs[8].get_team_elo(v))
    print(LCS_ELOs[9].get_team_elo(v))

print(
LCS_ELOs[8].LCSSpring_Playoffs_New(LCS2021_Spring_Teams["C9"],
                                   LCS2021_Spring_Teams["TSM"],
                                   LCS2021_Spring_Teams["TL"],
                                   LCS2021_Spring_Teams["100"],
                                   LCS2021_Spring_Teams["DIG"],
                                   LCS2021_Spring_Teams["EG"]))
print(
LCS_ELOs[9].LCSSpring_Playoffs_New(LCS2021_Spring_Teams["C9"],
                                   LCS2021_Spring_Teams["TSM"],
                                   LCS2021_Spring_Teams["TL"],
                                   LCS2021_Spring_Teams["100"],
                                   LCS2021_Spring_Teams["DIG"],
                                   LCS2021_Spring_Teams["EG"]))

print("\nLCS 2021 Summer")
LCS2021_Summer_Teams = {
    "100": ["Ssumday", "Closer", "Abbedagge", "FBI", "huhi"],
    "C9": ["Fudge", "Blaber", "Perkz", "Zven", "Vulcan"],
    "CLG": ["Finn", "Broxah", "Pobelter", "WildTurtle", "Smoothie"],
    "DIG": ["FakeGod", "Akaadian", "Yusui", "Neo", "aphromoo"],
    "EG": ["Impact", "Contractz", "Jiizuke", "Deftly", "IgNar"], # Svenskeren was the main Jungle for most of the regular season and played two individual games during playoffs, but Contractz took over the starting Jungle spot for playoffs
    "FLY": ["Licorice", "Josedeodo", "Palafox", "Johnsun", "Dreams"],
    "GG": ["Licorice", "wallflower", "Ablazeolive", "Stixxay", "Newbie"], # Licorice joined GG after being benched by FLY
    "IMT": ["Revenge", "Xerxe", "Insanity", "Raes", "Destiny"],
    "TL": ["Alphari", "Santorin", "Jensen", "Tactical", "CoreJJ"],
    "TSM": ["Huni", "Spica", "PowerOfEvil", "Lost", "SwordArt"],
}

for k, v in LCS2021_Summer_Teams.items():
    print(k)
    print(LCS_ELOs[10].get_team_elo(v))
    print(LCS_ELOs[11].get_team_elo(v))

print(
LCS_ELOs[10].LCSSummer_Playoffs_New(LCS2021_Summer_Teams["TSM"],
                                    LCS2021_Summer_Teams["100"],
                                    LCS2021_Summer_Teams["EG"],
                                    LCS2021_Summer_Teams["C9"],
                                    LCS2021_Summer_Teams["TL"],
                                    LCS2021_Summer_Teams["DIG"],
                                    LCS2021_Summer_Teams["IMT"],
                                    LCS2021_Summer_Teams["GG"]))
print(
LCS_ELOs[11].LCSSummer_Playoffs_New(LCS2021_Summer_Teams["TSM"],
                                    LCS2021_Summer_Teams["100"],
                                    LCS2021_Summer_Teams["EG"],
                                    LCS2021_Summer_Teams["C9"],
                                    LCS2021_Summer_Teams["TL"],
                                    LCS2021_Summer_Teams["DIG"],
                                    LCS2021_Summer_Teams["IMT"],
                                    LCS2021_Summer_Teams["GG"]))

print("\nLCS 2022 Spring")
LCS2022_Spring_Teams = {
    "100": ["Ssumday", "Closer", "Abbedagge", "FBI", "huhi"],
    "C9": ["Summit", "Blaber", "Fudge", "Berserker", "Winsome"], # Isles was subbed in for part of playoffs, but he does not have any games played up until this point, so his ELO is unknown
    "CLG": ["Jenkins", "Contractz", "Palafox", "Luger", "Poome"],
    "DIG": ["FakeGod", "River", "Blue", "Neo", "Biofrost"],
    "EG": ["Impact", "Inspired", "jojopyun", "Danny", "Vulcan"],
    "FLY": ["Kumo", "Josedeodo", "toucouille", "Johnsun", "aphromoo"],
    "GG": ["Licorice", "Pridestalkr", "Ablazeolive", "Lost", "Olleh"],
    "IMT": ["Revenge", "Xerxe", "PowerOfEvil", "WildTurtle", "Destiny"],
    "TL": ["Bwipo", "Santorin", "Bjergsen", "Hans Sama", "CoreJJ"],
    "TSM": ["Huni", "Spica", "Takeover", "Tactical", "Shenyi"],
}

for k, v in LCS2022_Spring_Teams.items():
    print(k)
    print(LCS_ELOs[12].get_team_elo(v))
    print(LCS_ELOs[13].get_team_elo(v))

print(
LCS_ELOs[12].LCSSpring_Playoffs_New(LCS2022_Spring_Teams["TL"],
                                    LCS2022_Spring_Teams["C9"],
                                    LCS2022_Spring_Teams["100"],
                                    LCS2022_Spring_Teams["EG"],
                                    LCS2022_Spring_Teams["FLY"],
                                    LCS2022_Spring_Teams["GG"]))
print(
LCS_ELOs[13].LCSSpring_Playoffs_New(LCS2022_Spring_Teams["TL"],
                                    LCS2022_Spring_Teams["C9"],
                                    LCS2022_Spring_Teams["100"],
                                    LCS2022_Spring_Teams["EG"],
                                    LCS2022_Spring_Teams["FLY"],
                                    LCS2022_Spring_Teams["GG"]))

print("\nLCS 2022 Summer")
LCS2022_Summer_Teams = {
    "100": ["Ssumday", "Closer", "Abbedagge", "FBI", "huhi"],
    "C9": ["Fudge", "Blaber", "Jensen", "Berserker", "Zven"],
    "CLG": ["Dhokla", "Contractz", "Palafox", "Luger", "Poome"],
    "DIG": ["Gamsu", "River", "Blue", "Spawn", "Biofrost"],
    "EG": ["Impact", "Inspired", "jojopyun", "Danny", "Vulcan"], # Kaori was subbed in for part of playoffs, but he does not have any games played up until this point, so his ELO is unknown
    "FLY": ["Philip", "Josedeodo", "toucouille", "Johnsun", "aphromoo"],
    "GG": ["Licorice", "River", "Ablazeolive", "Stixxay", "Olleh"], # River joined GG after being benched by DIG
    "IMT": ["Revenge", "Kenvi", "PowerOfEvil", "Lost", "IgNar"],
    "TL": ["Bwipo", "Santorin", "Bjergsen", "Hans Sama", "CoreJJ"],
    "TSM": ["Solo", "Spica", "Maple", "Tactical", "Chime"], # Instinct played two individual games during playoffs and was replaced by Tactical for the remainder of playoffs
}

for k, v in LCS2022_Summer_Teams.items():
    print(k)
    print(LCS_ELOs[14].get_team_elo(v))
    print(LCS_ELOs[15].get_team_elo(v))

print(
LCS_ELOs[14].LCSSummer_Playoffs_New(LCS2022_Summer_Teams["EG"],
                                    LCS2022_Summer_Teams["100"],
                                    LCS2022_Summer_Teams["TL"],
                                    LCS2022_Summer_Teams["CLG"],
                                    LCS2022_Summer_Teams["C9"],
                                    LCS2022_Summer_Teams["FLY"],
                                    LCS2022_Summer_Teams["TSM"],
                                    LCS2022_Summer_Teams["GG"]))
print(
LCS_ELOs[15].LCSSummer_Playoffs_New(LCS2022_Summer_Teams["EG"],
                                    LCS2022_Summer_Teams["100"],
                                    LCS2022_Summer_Teams["TL"],
                                    LCS2022_Summer_Teams["CLG"],
                                    LCS2022_Summer_Teams["C9"],
                                    LCS2022_Summer_Teams["FLY"],
                                    LCS2022_Summer_Teams["TSM"],
                                    LCS2022_Summer_Teams["GG"]))

print("\nLCS 2023 Spring")
LCS2023_Spring_Teams = {
    "100": ["Tenacity", "Closer", "Bjergsen", "Doublelift", "Busio"],
    "C9": ["Fudge", "Blaber", "EMENES", "Berserker", "Zven"],
    "CLG": ["Dhokla", "Contractz", "Palafox", "Luger", "Poome"],
    "DIG": ["Armut", "Santorin", "Jensen", "Tomo", "IgNar"],
    "EG": ["Ssumday", "Inspired", "jojopyun", "FBI", "Vulcan"],
    "FLY": ["Impact", "Spica", "VicLa", "Prince", "Eyla"], # Winsome played one series during playoffs
    "GG": ["Licorice", "River", "Gori", "Stixxay", "huhi"],
    "IMT": ["Revenge", "Kenvi", "Ablazeolive", "Tactical", "Fleshy"],
    "TL": ["Summit", "Pyosik", "Haeri", "Yeon", "CoreJJ"],
    "TSM": ["Solo", "Bugi", "Maple", "Neo", "Chime"],
}

for k, v in LCS2023_Spring_Teams.items():
    print(k)
    print(LCS_ELOs[16].get_team_elo(v))
    print(LCS_ELOs[17].get_team_elo(v))

print(
LCS_ELOs[16].LCSSpring_Playoffs_New(LCS2023_Spring_Teams["C9"],
                                    LCS2023_Spring_Teams["FLY"],
                                    LCS2023_Spring_Teams["100"],
                                    LCS2023_Spring_Teams["CLG"],
                                    LCS2023_Spring_Teams["EG"],
                                    LCS2023_Spring_Teams["GG"]))
print(
LCS_ELOs[17].LCSSpring_Playoffs_New(LCS2023_Spring_Teams["C9"],
                                    LCS2023_Spring_Teams["FLY"],
                                    LCS2023_Spring_Teams["100"],
                                    LCS2023_Spring_Teams["CLG"],
                                    LCS2023_Spring_Teams["EG"],
                                    LCS2023_Spring_Teams["GG"]))

print("\nLEC 2019 Spring")
LEC2019_Spring_Teams = {
    "XL": ["Expect", "Caedrel", "Special", "Jeskla", "Kasing"],
    "S04": ["Odoamne", "Memento", "Abbedagge", "Upset", "IgNar"],
    "FNC": ["Bwipo", "Broxah", "Nemesis", "Rekkles", "Hylissang"],
    "G2": ["Wunder", "Jankos", "Caps", "Perkz", "Mikyx"],
    "MSF": ["sOAZ", "Maxlore", "FEBIVEN", "Hans Sama", "GorillA"],
    "OG": ["Alphari", "Kold", "Nukeduck", "Patrik", "Mithy"],
    "RGE": ["Profit", "Kikis", "Sencux", "HeaQ", "Wadid"],
    "SK": ["Werlyb", "Selfmade", "Pirean", "Crownie", "Dreams"],
    "SPY": ["Vizicsacsi", "Xerxe", "Humanoid", "Kobbe", "Tore"],
    "VIT": ["Cabochard", "Mowgli", "Jiizuke", "Attila", "Jactroll"]
}

for k, v in LEC2019_Spring_Teams.items():
    print(k)
    print(LEC_ELOs[0].get_team_elo(v))
    print(LEC_ELOs[1].get_team_elo(v))

LEC_ELOs[0].LEC_Playoffs_Old(LEC2019_Spring_Teams["G2"],
                             LEC2019_Spring_Teams["OG"],
                             LEC2019_Spring_Teams["FNC"],
                             LEC2019_Spring_Teams["SPY"],
                             LEC2019_Spring_Teams["SK"],
                             LEC2019_Spring_Teams["VIT"])
LEC_ELOs[1].LEC_Playoffs_Old(LEC2019_Spring_Teams["G2"],
                             LEC2019_Spring_Teams["OG"],
                             LEC2019_Spring_Teams["FNC"],
                             LEC2019_Spring_Teams["SPY"],
                             LEC2019_Spring_Teams["SK"],
                             LEC2019_Spring_Teams["VIT"])

print("\nLEC 2019 Summer")
LEC2019_Summer_Teams = {
    "XL": ["Expect", "Caedrel", "Mickey", "Jeskla", "Mystiques"],
    "S04": ["Odoamne", "Trick", "Abbedagge", "Upset", "IgNar"],
    "FNC": ["Bwipo", "Broxah", "Nemesis", "Rekkles", "Hylissang"],
    "G2": ["Wunder", "Jankos", "Caps", "Perkz", "Mikyx"],
    "MSF": ["Dan Dan", "Kirei", "LIDER", "Hans Sama", "Hiiva"],
    "OG": ["Alphari", "Kold", "Nukeduck", "Patrik", "Mithy"],
    "RGE": ["Finn", "Inspired", "Larssen", "Woolite", "Vander"],
    "SK": ["Sacre", "Selfmade", "Pirean", "Crownie", "Dreams"],
    "SPY": ["Vizicsacsi", "Xerxe", "Humanoid", "Kobbe", "Tore"],
    "VIT": ["Cabochard", "Mowgli", "Jiizuke", "Attila", "Jactroll"]
}

for k, v in LEC2019_Summer_Teams.items():
    print(k)
    print(LEC_ELOs[2].get_team_elo(v))
    print(LEC_ELOs[3].get_team_elo(v))

LEC_ELOs[2].LEC_Playoffs_Old(LEC2019_Summer_Teams["G2"],
                             LEC2019_Summer_Teams["OG"],
                             LEC2019_Summer_Teams["SPY"],
                             LEC2019_Summer_Teams["S04"],
                             LEC2019_Summer_Teams["VIT"],
                             LEC2019_Summer_Teams["RGE"])
LEC_ELOs[3].LEC_Playoffs_Old(LEC2019_Summer_Teams["G2"],
                             LEC2019_Summer_Teams["OG"],
                             LEC2019_Summer_Teams["SPY"],
                             LEC2019_Summer_Teams["S04"],
                             LEC2019_Summer_Teams["VIT"],
                             LEC2019_Summer_Teams["RGE"])

print("\nLEC 2020 Spring")
LEC2020_Spring_Teams = {
    "XL": ["Expect", "Caedrel", "Mickey", "Patrik", "Tore"],
    "S04": ["Odoamne", "Lurox", "Abbedagge", "Innaxe", "Dreams"],
    "FNC": ["Bwipo", "Selfmade", "Nemesis", "Rekkles", "Hylissang"],
    "G2": ["Wunder", "Jankos", "Perkz", "Caps", "Mikyx"],
    "MAD": ["Orome", "shad0w", "Humanoid", "Carzzy", "Kaiser"],
    "MSF": ["Dan Dan", "Razork", "FEBIVEN", "Bvoy", "denyk"],
    "OG": ["Alphari", "Xerxe", "Nukeduck", "Upset", "Destiny"],
    "RGE": ["Finn", "Inspired", "Larssen", "Hans Sama", "Vander"],
    "SK": ["Sacre", "Trick", "JNX", "Crownie", "LIMIT"],
    "VIT": ["Cabochard", "Skeanz", "Saken", "Comp", "Jactroll"]
}

for k, v in LEC2020_Spring_Teams.items():
    print(k)
    print(LEC_ELOs[4].get_team_elo(v))
    print(LEC_ELOs[5].get_team_elo(v))

LEC_ELOs[4].LEC_Playoffs_Gauntlet(LEC2020_Spring_Teams["G2"],
                                  LEC2020_Spring_Teams["FNC"],
                                  LEC2020_Spring_Teams["OG"],
                                  LEC2020_Spring_Teams["MAD"],
                                  LEC2020_Spring_Teams["MSF"],
                                  LEC2020_Spring_Teams["RGE"])
LEC_ELOs[5].LEC_Playoffs_Gauntlet(LEC2020_Spring_Teams["G2"],
                                  LEC2020_Spring_Teams["FNC"],
                                  LEC2020_Spring_Teams["OG"],
                                  LEC2020_Spring_Teams["MAD"],
                                  LEC2020_Spring_Teams["MSF"],
                                  LEC2020_Spring_Teams["RGE"])


print("\nLEC 2020 Summer")
LEC2020_Summer_Teams = {
    "XL": ["Kryze", "Caedrel", "Mickey", "Patrik", "Tore"],
    "S04": ["Odoamne", "Gilius", "Abbedagge", "Neon", "Dreams"],
    "FNC": ["Bwipo", "Selfmade", "Nemesis", "Rekkles", "Hylissang"],
    "G2": ["Wunder", "Jankos", "Caps", "Perkz", "Mikyx"],
    "MAD": ["Orome", "shad0w", "Humanoid", "Carzzy", "Kaiser"],
    "MSF": ["Dan Dan", "Razork", "FEBIVEN", "Kobbe", "Doss"],
    "OG": ["Alphari", "Xerxe", "Nukeduck", "Upset", "Jactroll"],
    "RGE": ["Finn", "Inspired", "Larssen", "Hans Sama", "Vander"],
    "SK": ["JNX", "Trick", "ZaZee", "Crownie", "LIMIT"],
    "VIT": ["Cabochard", "Skeanz", "Milica", "Comp", "Labrov"]
}

for k, v in LEC2020_Summer_Teams.items():
    print(k)
    print(LEC_ELOs[6].get_team_elo(v))
    print(LEC_ELOs[7].get_team_elo(v))

LEC_ELOs[6].LEC_Playoffs_Gauntlet(LEC2020_Summer_Teams["RGE"],
                                  LEC2020_Summer_Teams["MAD"],
                                  LEC2020_Summer_Teams["G2"],
                                  LEC2020_Summer_Teams["FNC"],
                                  LEC2020_Summer_Teams["SK"],
                                  LEC2020_Summer_Teams["S04"])
LEC_ELOs[7].LEC_Playoffs_Gauntlet(LEC2020_Summer_Teams["RGE"],
                                  LEC2020_Summer_Teams["MAD"],
                                  LEC2020_Summer_Teams["G2"],
                                  LEC2020_Summer_Teams["FNC"],
                                  LEC2020_Summer_Teams["SK"],
                                  LEC2020_Summer_Teams["S04"])
    
print("\nLEC 2021 Spring")
LEC2021_Spring_Teams = {
    "AST": ["WhiteKnight", "Zanzarah", "MagiFelix", "Jeskla", "promisq"],
    "XL": ["Kryze", "Dan", "Czekolad", "Patrik", "Tore"],
    "S04": ["BrokenBlade", "Gilius", "Abbedagge", "Neon", "LIMIT"],
    "FNC": ["Bwipo", "Selfmade", "Nisqy", "Upset", "Hylissang"],
    "G2": ["Wunder", "Jankos", "Caps", "Rekkles", "Mikyx"],
    "MAD": ["Armut", "Elyoya", "Humanoid", "Carzzy", "Kaiser"],
    "MSF": ["HiRit", "Razork", "Vetheo", "Kobbe", "Vander"],
    "RGE": ["Odoamne", "Inspired", "Larssen", "Hans Sama", "Trymbi"],
    "SK": ["JNX", "TynX", "Blue", "Jezu", "Treatz"],
    "VIT": ["Szygenda", "Skeanz", "Milica", "Crownie", "Labrov"]
}

for k, v in LEC2021_Spring_Teams.items():
    print(k)
    print(LEC_ELOs[8].get_team_elo(v))
    print(LEC_ELOs[9].get_team_elo(v))

print("\nLEC 2021 Summer")
LEC2021_Summer_Teams = {
    "AST": ["WhiteKnight", "Zanzarah", "MagiFelix", "Jeskla", "promisq"],
    "XL": ["Kryze", "Markoon", "Czekolad", "Patrik", "Advienne"],
    "S04": ["BrokenBlade", "Kirei", "nuc", "Neon", "LIMIT"],
    "FNC": ["Adam", "Bwipo", "Nisqy", "Upset", "Hylissang"],
    "G2": ["Wunder", "Jankos", "Caps", "Rekkles", "Mikyx"],
    "MAD": ["Armut", "Elyoya", "Humanoid", "Carzzy", "Kaiser"],
    "MSF": ["HiRit", "Razork", "Vetheo", "Kobbe", "Vander"],
    "RGE": ["Odoamne", "Inspired", "Larssen", "Hans Sama", "Trymbi"],
    "SK": ["JNX", "Treatz", "Blue", "Jezu", "Lilipp"],
    "VIT": ["Szygenda", "Selfmade", "LIDER", "Crownie", "Labrov"]
}

for k, v in LEC2021_Summer_Teams.items():
    print(k)
    print(LEC_ELOs[10].get_team_elo(v))
    print(LEC_ELOs[11].get_team_elo(v))

print("\nLEC 2022 Spring")
LEC2022_Spring_Teams = {
    "AST": ["WhiteKnight", "Zanzarah", "Dajor", "Kobbe", "promisq"],
    "XL": ["Finn", "Markoon", "Nukeduck", "Patrik", "Mikyx"],
    "FNC": ["Wunder", "Razork", "Humanoid", "Upset", "Hylissang"],
    "G2": ["BrokenBlade", "Jankos", "Caps", "Flakked", "Targamas"],
    "MAD": ["Armut", "Elyoya", "Reeker", "UNF0RGIVEN", "Kaiser"],
    "MSF": ["HiRit", "Shlatan", "Vetheo", "Neon", "Mersa"],
    "RGE": ["Odoamne", "Malrang", "Larssen", "Comp", "Trymbi"],
    "SK": ["JNX", "Gilius", "Sertuss", "Jezu", "Treatz"],
    "BDS": ["Adam", "Cinkrof", "nuc", "xMatty", "LIMIT"],
    "VIT": ["Alphari", "Selfmade", "Perkz", "Carzzy", "Labrov"]
}

for k, v in LEC2022_Spring_Teams.items():
    print(k)
    print(LEC_ELOs[12].get_team_elo(v))
    print(LEC_ELOs[13].get_team_elo(v))

print("\nLEC 2022 Summer")
LEC2022_Summer_Teams = {
    "AST": ["Vizicsacsi", "Xerxe", "Dajor", "Kobbe", "JeongHoon"],
    "XL": ["Finn", "Markoon", "Nukeduck", "Patrik", "Mikyx"],
    "FNC": ["Wunder", "Razork", "Humanoid", "Upset", "Hylissang"],
    "G2": ["BrokenBlade", "Jankos", "Caps", "Flakked", "Targamas"],
    "MAD": ["Armut", "Elyoya", "Nisqy", "UNF0RGIVEN", "Kaiser"],
    "MSF": ["Irrelevant", "Zanzarah", "Vetheo", "Neon", "Mersa"],
    "RGE": ["Odoamne", "Malrang", "Larssen", "Comp", "Trymbi"],
    "SK": ["JNX", "Gilius", "Sertuss", "Jezu", "Treatz"],
    "BDS": ["Adam", "Cinkrof", "nuc", "xMatty", "Erdote"],
    "VIT": ["Alphari", "Haru", "Perkz", "Carzzy", "Labrov"]
}

for k, v in LEC2022_Summer_Teams.items():
    print(k)
    print(LEC_ELOs[14].get_team_elo(v))
    print(LEC_ELOs[15].get_team_elo(v))

print("\nLEC 2023 Winter")
LEC2023_Winter_Teams = {
    "AST": ["Finn", "113", "Dajor", "Kobbe", "JeongHoon"],
    "XL": ["Odoamne", "Xerxe", "Vetheo", "Patrik", "Targamas"],
    "FNC": ["Wunder", "Razork", "Humanoid", "Rekkles", "Rhuckz"],
    "G2": ["BrokenBlade", "Yike", "Caps", "Hans Sama", "Mikyx"],
    "KOI": ["Szygenda", "Malrang", "Larssen", "Comp", "Trymbi"],
    "MAD": ["Chasy", "Elyoya", "Nisqy", "Carzzy", "Hylissang"],
    "SK": ["Irrelevant", "Markoon", "Sertuss", "Exakick", "Doss"],
    "BDS": ["Adam", "Sheo", "nuc", "Crownie", "Labrov"],
    "TH": ["Evi", "Jankos", "Ruby", "Jackspektra", "Mersa"],
    "VIT": ["Photon", "Bo", "Perkz", "Neon", "Kaiser"]
}

for k, v in LEC2023_Winter_Teams.items():
    print(k)
    print(LEC_ELOs[16].get_team_elo(v))
    print(LEC_ELOs[17].get_team_elo(v))

LEC_ELOs[16].LEC_Playoffs_New(LEC2023_Winter_Teams["G2"],
                              LEC2023_Winter_Teams["KOI"],
                              LEC2023_Winter_Teams["SK"],
                              LEC2023_Winter_Teams["MAD"])
LEC_ELOs[17].LEC_Playoffs_New(LEC2023_Winter_Teams["G2"],
                              LEC2023_Winter_Teams["KOI"],
                              LEC2023_Winter_Teams["SK"],
                              LEC2023_Winter_Teams["MAD"])

print("\nLEC 2023 Spring")
LEC2023_Spring_Teams = {
    "AST": ["Finn", "113", "LIDER", "Kobbe", "JeongHoon"],
    "XL": ["Odoamne", "Xerxe", "Vetheo", "Patrik", "LIMIT"],
    "FNC": ["Oscarinin", "Razork", "Humanoid", "Rekkles", "Advienne"],
    "G2": ["BrokenBlade", "Yike", "Caps", "Hans Sama", "Mikyx"],
    "KOI": ["Szygenda", "Malrang", "Larssen", "Comp", "Trymbi"],
    "MAD": ["Chasy", "Elyoya", "Nisqy", "Carzzy", "Hylissang"],
    "SK": ["Irrelevant", "Markoon", "Sertuss", "Exakick", "Doss"],
    "BDS": ["Adam", "Sheo", "nuc", "Crownie", "Labrov"],
    "TH": ["Evi", "Jankos", "Ruby", "Jackspektra", "Mersa"],
    "VIT": ["Photon", "Bo", "Perkz", "Upset", "Kaiser"]
}

for k, v in LEC2023_Spring_Teams.items():
    print(k)
    print(LEC_ELOs[18].get_team_elo(v))
    print(LEC_ELOs[19].get_team_elo(v))

LEC_ELOs[18].LEC_Playoffs_New(LEC2023_Spring_Teams["VIT"],
                              LEC2023_Spring_Teams["BDS"],
                              LEC2023_Spring_Teams["MAD"],
                              LEC2023_Spring_Teams["G2"])
LEC_ELOs[19].LEC_Playoffs_New(LEC2023_Spring_Teams["VIT"],
                              LEC2023_Spring_Teams["BDS"],
                              LEC2023_Spring_Teams["MAD"],
                              LEC2023_Spring_Teams["G2"])

print("\nLCK 2019 Spring")
LCK2019_Spring_Teams = {
    "AF": ["Kiin", "Dread", "ucal", "Aiming", "Jelly"],
    "DWG": ["Nuguri", "Canyon", "ShowMaker", "Nuclear", "BeryL"], # Hoit played one individual game during playoffs
    "GEN": ["CuVee", "Peanut", "Fly", "Ruler", "Life"],
    "GRF": ["Sw0rd", "Tarzan", "Chovy", "Viper", "Lehends"],
    "HLE": ["Thal", "bonO", "Lava", "Sangyoon", "Key"],
    "JAG": ["Lindarang", "Malrang", "Yaharong", "Route", "Kellin"],
    "KZ": ["Rascal", "Cuzz", "PawN", "Deft", "TusiN"],
    "KT": ["Smeb", "Score", "Bdd", "Zenit", "SnowFlower"],
    "SB": ["Summit", "OnFleek", "Dove", "Ghost", "Joker"],
    "SKT": ["Khan", "Clid", "Faker", "Teddy", "Mata"]
}

for k, v in LCK2019_Spring_Teams.items():
    print(k)
    print(LCK_ELOs[0].get_team_elo(v))
    print(LCK_ELOs[1].get_team_elo(v))

LCK_ELOs[0].LCK_Playoffs_Old(LCK2019_Spring_Teams["GRF"],
                             LCK2019_Spring_Teams["SKT"],
                             LCK2019_Spring_Teams["KZ"],
                             LCK2019_Spring_Teams["DWG"],
                             LCK2019_Spring_Teams["SB"])
LCK_ELOs[1].LCK_Playoffs_Old(LCK2019_Spring_Teams["GRF"],
                             LCK2019_Spring_Teams["SKT"],
                             LCK2019_Spring_Teams["KZ"],
                             LCK2019_Spring_Teams["DWG"],
                             LCK2019_Spring_Teams["SB"])


print("\nLCK 2019 Summer")
LCK2019_Summer_Teams = {
    "AF": ["Kiin", "Dread", "ucal", "Aiming", "Test Sup"],
    "DWG": ["Nuguri", "Canyon", "ShowMaker", "Nuclear", "BeryL"],
    "GEN": ["CuVee", "Peanut", "Fly", "Ruler", "Life"],
    "GRF": ["Doran", "Tarzan", "Chovy", "Viper", "Lehends"],
    "HLE": ["SoHwan", "bonO", "Lava", "Sangyoon", "Key"],
    "JAG": ["Lindarang", "Malrang", "Yaharong", "Route", "Kellin"],
    "KZ": ["Rascal", "Cuzz", "Naehyun", "Deft", "TusiN"],
    "KT": ["Kingen", "Score", "Bdd", "PraY", "SnowFlower"],
    "SB": ["Summit", "OnFleek", "Dove", "Ghost", "Joker"],
    "SKT": ["Khan", "Clid", "Faker", "Teddy", "Effort"]
}

for k, v in LCK2019_Summer_Teams.items():
    print(k)
    print(LCK_ELOs[2].get_team_elo(v))
    print(LCK_ELOs[3].get_team_elo(v))

LCK_ELOs[2].LCK_Playoffs_Old(LCK2019_Summer_Teams["GRF"],
                             LCK2019_Summer_Teams["DWG"],
                             LCK2019_Summer_Teams["SB"],
                             LCK2019_Summer_Teams["SKT"],
                             LCK2019_Summer_Teams["AF"])
LCK_ELOs[3].LCK_Playoffs_Old(LCK2019_Summer_Teams["GRF"],
                             LCK2019_Summer_Teams["DWG"],
                             LCK2019_Summer_Teams["SB"],
                             LCK2019_Summer_Teams["SKT"],
                             LCK2019_Summer_Teams["AF"])

print("\nLCK 2020 Spring")
LCK2020_Spring_Teams = {
    "AF": ["Kiin", "Spirit", "Fly", "Mystic", "Jelly"],
    "APK": ["ikssu", "Flawless", "Cover", "HyBriD", "Secret"],
    "DWG": ["Nuguri", "Canyon", "ShowMaker", "Ghost", "BeryL"],
    "DRX": ["Doran", "Pyosik", "Chovy", "Deft", "Keria"],
    "GEN": ["Rascal", "Clid", "Bdd", "Ruler", "Life"],
    "GRF": ["Sw0rd", "Tarzan", "ucal", "Viper", "Kabbie"],
    "HLE": ["CuVee", "Haru", "Tempt", "Vsta", "Lehends"],
    "KT": ["SoHwan", "bonO", "Kuro", "Aiming", "TusiN"],
    "SB": ["Summit", "OnFleek", "Dove", "Route", "Joker"],
    "T1": ["Canna", "Cuzz", "Faker", "Teddy", "Effort"]
}

for k, v in LCK2020_Spring_Teams.items():
    print(k)
    print(LCK_ELOs[4].get_team_elo(v))
    print(LCK_ELOs[5].get_team_elo(v))

print("\nLCK 2020 Summer")
LCK2020_Summer_Teams = {
    "AF": ["Kiin", "Spirit", "Fly", "Mystic", "Ben"],
    "DWG": ["Nuguri", "Canyon", "ShowMaker", "Ghost", "BeryL"],
    "DRX": ["Doran", "Pyosik", "Chovy", "Deft", "Keria"],
    "GEN": ["Rascal", "Clid", "Bdd", "Ruler", "Life"],
    "HLE": ["DuDu", "Haru", "Mireu", "Viper", "Lehends"],
    "KT": ["Smeb", "bonO", "Kuro", "Aiming", "TusiN"],
    "SB": ["Summit", "OnFleek", "FATE", "Route", "GorillA"],
    "SP": ["ikssu", "Flawless", "Mickey", "HyBriD", "Secret"],
    "T1": ["Canna", "Cuzz", "Faker", "Teddy", "Effort"], # Clozer played one individual game during playoffs
    "TD": ["Rich", "Beyond", "Kuzan", "deokdam", "GuGer"]
}

for k, v in LCK2020_Summer_Teams.items():
    print(k)
    print(LCK_ELOs[6].get_team_elo(v))
    print(LCK_ELOs[7].get_team_elo(v))

print("\nLCK 2021 Spring")
LCK2021_Spring_Teams = {
    "AF": ["Kiin", "Dread", "Fly", "Bang", "Lehends"],
    "DRX": ["Kingen", "Pyosik", "Quad", "BAO", "Pleata"],
    "DK": ["Khan", "Canyon", "ShowMaker", "Ghost", "BeryL"],
    "BRO": ["HOYA", "UmTi", "Lava", "Hena", "Delight"],
    "GEN": ["Rascal", "Clid", "Bdd", "Ruler", "Life"],
    "HLE": ["Morgan", "Winter 7", "Chovy", "Deft", "Vsta"], # Mir played three individual games during playoffs
    "KT": ["Doran", "gideon", "ucal", "HyBriD", "Zzus"],
    "LSB": ["Summit", "Croco", "FATE", "Prince", "Effort"],
    "NS": ["Rich", "Peanut", "Bay", "deokdam", "Kellin"],
    "T1": ["Canna", "Cuzz", "Faker", "Teddy", "Keria"]
}

for k, v in LCK2021_Spring_Teams.items():
    print(k)
    print(LCK_ELOs[8].get_team_elo(v))
    print(LCK_ELOs[9].get_team_elo(v))

print("\nLCK 2021 Summer")
LCK2021_Summer_Teams = {
    "AF": ["Kiin", "Dread", "Fly", "Leo", "Lehends"],
    "DRX": ["Kingen", "Pyosik", "Quad", "Taeyoon", "Jun"],
    "DK": ["Khan", "Canyon", "ShowMaker", "Ghost", "BeryL"],
    "BRO": ["HOYA", "UmTi", "Lava", "Hena", "Delight"],
    "GEN": ["Rascal", "Clid", "Bdd", "Ruler", "Life"],
    "HLE": ["Morgan", "Willer", "Chovy", "Deft", "Vsta"],
    "KT": ["Doran", "Blank", "Dove", "Noah", "Harp"],
    "LSB": ["Summit", "Croco", "FATE", "Prince", "Effort"],
    "NS": ["Rich", "Peanut", "Gori", "deokdam", "Kellin"],
    "T1": ["Canna", "Oner", "Faker", "Teddy", "Keria"] # Gumayusi played two individual games in playoffs
}

for k, v in LCK2021_Summer_Teams.items():
    print(k)
    print(LCK_ELOs[10].get_team_elo(v))
    print(LCK_ELOs[11].get_team_elo(v))

print("\nLCK 2022 Spring")
LCK2022_Spring_Teams = {
    "DRX": ["Kingen", "Pyosik", "Zeka", "Deft", "BeryL"],
    "DK": ["Burdol", "Canyon", "ShowMaker", "deokdam", "Kellin"],
    "BRO": ["Morgan", "UmTi", "Lava", "Hena", "Delight"],
    "GEN": ["Doran", "Peanut", "Chovy", "Ruler", "Lehends"],
    "HLE": ["DuDu", "OnFleek", "Karis", "Hans SamD", "Vsta"],
    "KT": ["Rascal", "Cuzz", "Aria", "Aiming", "Life"],
    "KDF": ["Kiin", "Ellim", "FATE", "Teddy", "Hoit"],
    "LSB": ["Dove", "Croco", "Clozer", "Ice", "Kael"],
    "NS": ["Canna", "Dread", "Bdd", "Ghost", "Peter"],
    "T1": ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]
}

for k, v in LCK2022_Spring_Teams.items():
    print(k)
    print(LCK_ELOs[12].get_team_elo(v))
    print(LCK_ELOs[13].get_team_elo(v))

print("\nLCK 2022 Summer")
LCK2022_Summer_Teams = {
    "DRX": ["Kingen", "Pyosik", "Zeka", "Deft", "BeryL"],
    "DK": ["Nuguri", "Canyon", "ShowMaker", "deokdam", "Kellin"], # Nuguri played the entire regular season but missed most of playoffs, this is an exception to the above rule
    "BRO": ["Morgan", "UmTi", "Lava", "Hena", "Delight"],
    "GEN": ["Doran", "Peanut", "Chovy", "Ruler", "Lehends"],
    "HLE": ["DuDu", "OnFleek", "Karis", "Hans SamD", "Vsta"],
    "KT": ["Rascal", "Cuzz", "VicLa", "Aiming", "Life"],
    "KDF": ["Kiin", "Ellim", "FATE", "Teddy", "Hoit"],
    "LSB": ["Dove", "Croco", "Clozer", "Prince", "Kael"],
    "NS": ["Canna", "Dread", "Bdd", "Ghost", "Effort"],
    "T1": ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]
}

for k, v in LCK2022_Summer_Teams.items():
    print(k)
    print(LCK_ELOs[14].get_team_elo(v))
    print(LCK_ELOs[15].get_team_elo(v))

print("\nLCK 2023 Spring")
LCK2023_Spring_Teams = {
    "BRO": ["Morgan", "UmTi", "Lava", "Hena", "Effort"],
    "DK": ["Canna", "Canyon", "ShowMaker", "Deft", "Kellin"],
    "DRX": ["Rascal", "Croco", "FATE", "deokdam", "Pleata"],
    "GEN": ["Doran", "Peanut", "Chovy", "Peyz", "Lehends"],
    "HLE": ["Kingen", "Clid", "Zeka", "Viper", "Life"],
    "KT": ["Kiin", "Cuzz", "Bdd", "Aiming", "Lehends"],
    "KDF": ["DuDu", "YoungJae", "BuLLDoG", "Taeyoon", "Jun"],
    "LSB": ["Burdol", "Willer", "Clozer", "Envyy", "Kael"],
    "NS": ["DnDn", "Sylvie", "FIESTA", "vital", "Peter"],
    "T1": ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]
}

for k, v in LCK2023_Spring_Teams.items():
    print(k)
    print(LCK_ELOs[16].get_team_elo(v))
    print(LCK_ELOs[17].get_team_elo(v))

print("\nLPL 2021 Summer")
LPL2021_Summer_Teams = {
    "BLG": ["Biubiu", "Weiwei", "Zeka", "Aiming", "ppgod"],
    "EDG": ["Flandre", "Jiejie", "Scout", "Viper", "Meiko"], # JunJia played three individual games during playoffs
    "FPX": ["Nuguri", "Tian", "Doinb", "Lwx", "Crisp"],
    "IG": ["TheShy", "XUN", "Rookie", "Wink", "XinLiu"], # neny played the first eight series of the regular season, and TheShy played the last eight, neny played one more game than TheShy
    "JDG": ["Zoom", "Kanavi", "Yagao", "LokeN", "LvMao"],
    "LGD": ["fearness", "shad0w", "xiye", "Kramer", "Mark"],
    "LNG": ["Ale", "Tarzan", "icon", "Light", "Iwandy"],
    "OMG": ["New", "Aki", "Creme", "Able", "COLD"],
    "RA": ["Cube", "Leyan", "FoFo", "iBoy", "Hang"],
    "RW": ["Zdz", "Xiaohao", "Forge", "Betty", "QiuQiu"],
    "RNG": ["Xiaohu", "Wei", "Cryin", "GALA", "Ming"],
    "SN": ["Bin", "SofM", "Angel", "huanfeng", "ON"],
    "WE": ["Breathe", "beishang", "Shanks", "Elk", "Missing"],
    "TT": ["Langx", "Xiaopeng", "YeG", "Hans SamD", "yaoyao"],
    "TES": ["369", "Karsa", "knight", "JackeyLove", "Zhuo"],
    "UP": ["zs", "H4cker", "xiaocaobao", "Smlz", "ShiauC"],
    "V5": ["Aliez", "pzx", "ubao", "Kepler", "Missia"]
}

for k, v in LPL2021_Summer_Teams.items():
    print(k)
    print(LPL_ELOs[0].get_team_elo(v))
    print(LPL_ELOs[1].get_team_elo(v))

print(
LPL_ELOs[0].LPL_Playoffs(LPL2021_Summer_Teams["FPX"],
                         LPL2021_Summer_Teams["EDG"],
                         LPL2021_Summer_Teams["RA"],
                         LPL2021_Summer_Teams["RNG"],
                         LPL2021_Summer_Teams["TES"],
                         LPL2021_Summer_Teams["BLG"],
                         LPL2021_Summer_Teams["WE"],
                         LPL2021_Summer_Teams["LNG"],
                         LPL2021_Summer_Teams["SN"],
                         LPL2021_Summer_Teams["OMG"],))
print(
LPL_ELOs[1].LPL_Playoffs(LPL2021_Summer_Teams["FPX"],
                         LPL2021_Summer_Teams["EDG"],
                         LPL2021_Summer_Teams["RA"],
                         LPL2021_Summer_Teams["RNG"],
                         LPL2021_Summer_Teams["TES"],
                         LPL2021_Summer_Teams["BLG"],
                         LPL2021_Summer_Teams["WE"],
                         LPL2021_Summer_Teams["LNG"],
                         LPL2021_Summer_Teams["SN"],
                         LPL2021_Summer_Teams["OMG"],))


print("\nLPL 2022 Spring")
LPL2022_Spring_Teams = {
    "AL": ["Zdz", "Xiaohao", "Forge", "Betty", "QiuQiu"],
    "BLG": ["Breathe", "Weiwei", "FoFo", "Doggo", "Crisp"],
    "EDG": ["Flandre", "Jiejie", "Scout", "Viper", "Meiko"],
    "FPX": ["Xiaolaohu", "Clid", "Gori", "Lwx", "Hang"],
    "IG": ["Zika", "XUN", "Yuekai", "Wink", "XinLiu"],
    "JDG": ["369", "Kanavi", "Yagao", "Hope", "Missing"],
    "LGD": ["fearness", "shad0w", "Jay", "Eric", "Jinjiao"],
    "LNG": ["Ale", "Tarzan", "Doinb", "Light", "Iwandy"],
    "OMG": ["shanji", "Aki", "Creme", "Able", "COLD"],
    "RA": ["Cube", "Leyan", "Strive", "iBoy", "yuyanjia"],
    "RNG": ["Bin", "Wei", "Xiaohu", "GALA", "Ming"],
    "WE": ["Biubiu", "beishang", "Shanks", "Xing", "Kedaya"],
    "TT": ["New", "Chieftain", "ucal", "Puff", "Southwind"],
    "TES": ["Wayward", "Tian", "knight", "JackeyLove", "Mark"],
    "UP": ["zs", "H4cker", "Cryin", "Elk", "ShiauC"],
    "V5": ["Rich", "Karsa", "Rookie", "Photic", "ppgod"],
    "WBG": ["TheShy", "SofM", "Angel", "huanfeng", "ON"],
}

for k, v in LPL2022_Spring_Teams.items():
    print(k)
    print(LPL_ELOs[2].get_team_elo(v))
    print(LPL_ELOs[3].get_team_elo(v))

print(
LPL_ELOs[2].LPL_Playoffs(LPL2022_Spring_Teams["V5"],
                         LPL2022_Spring_Teams["RNG"],
                         LPL2022_Spring_Teams["JDG"],
                         LPL2022_Spring_Teams["LNG"],
                         LPL2022_Spring_Teams["TES"],
                         LPL2022_Spring_Teams["WBG"],
                         LPL2022_Spring_Teams["EDG"],
                         LPL2022_Spring_Teams["BLG"],
                         LPL2022_Spring_Teams["RA"],
                         LPL2022_Spring_Teams["FPX"]))
print(
LPL_ELOs[3].LPL_Playoffs(LPL2022_Spring_Teams["V5"],
                         LPL2022_Spring_Teams["RNG"],
                         LPL2022_Spring_Teams["JDG"],
                         LPL2022_Spring_Teams["LNG"],
                         LPL2022_Spring_Teams["TES"],
                         LPL2022_Spring_Teams["WBG"],
                         LPL2022_Spring_Teams["EDG"],
                         LPL2022_Spring_Teams["BLG"],
                         LPL2022_Spring_Teams["RA"],
                         LPL2022_Spring_Teams["FPX"]))

print("\nLPL 2022 Summer")
LPL2022_Summer_Teams = {
    "AL": ["Zdz", "Xiaohao", "Forge", "Betty", "QiuQiu"],
    "BLG": ["Bin", "Weiwei", "icon", "Doggo", "Crisp"],
    "EDG": ["Flandre", "Jiejie", "Scout", "Viper", "Meiko"], # JunJia played one individual series during playoffs
    "FPX": ["Summit", "Clid", "Care", "Lwx", "Hang"],
    "IG": ["Zika", "XUN", "Mole", "Ahn", "Wink"],
    "JDG": ["369", "Kanavi", "Yagao", "Hope", "Missing"],
    "LGD": ["fearness", "shad0w", "YeG", "Assum", "Jinjiao"],
    "LNG": ["Ale", "Tarzan", "Doinb", "Light", "LvMao"],
    "OMG": ["shanji", "Aki", "Creme", "Able", "COLD"],
    "RA": ["Cube", "Leyan", "Strive", "iBoy", "yuyanjia"],
    "RNG": ["Breathe", "Wei", "Xiaohu", "GALA", "Ming"],
    "WE": ["Biubiu", "beishang", "Shanks", "Xing", "Kedaya"],
    "TT": ["HOYA", "Beichuan", "ucal", "Kepler", "yaoyao"],
    "TES": ["Wayward", "Tian", "knight", "JackeyLove", "Mark"], # Qingtian played one individual game during playoffs
    "UP": ["Zoom", "H4cker", "Cryin", "Elk", "ShiauC"],
    "V5": ["Rich", "XLB", "Rookie", "Photic", "ppgod"], # Karsa played two individual games during playoffs, XLB played three
    "WBG": ["TheShy", "SofM", "Angel", "huanfeng", "ON"],
}

for k, v in LPL2022_Summer_Teams.items():
    print(k)
    print(LPL_ELOs[4].get_team_elo(v))
    print(LPL_ELOs[5].get_team_elo(v))

print(
LPL_ELOs[4].LPL_Playoffs(LPL2022_Summer_Teams["TES"],
                         LPL2022_Summer_Teams["JDG"],
                         LPL2022_Summer_Teams["V5"],
                         LPL2022_Summer_Teams["RNG"],
                         LPL2022_Summer_Teams["EDG"],
                         LPL2022_Summer_Teams["WBG"],
                         LPL2022_Summer_Teams["LNG"],
                         LPL2022_Summer_Teams["OMG"],
                         LPL2022_Summer_Teams["FPX"],
                         LPL2022_Summer_Teams["BLG"]))
print(
LPL_ELOs[5].LPL_Playoffs(LPL2022_Summer_Teams["TES"],
                         LPL2022_Summer_Teams["JDG"],
                         LPL2022_Summer_Teams["V5"],
                         LPL2022_Summer_Teams["RNG"],
                         LPL2022_Summer_Teams["EDG"],
                         LPL2022_Summer_Teams["WBG"],
                         LPL2022_Summer_Teams["LNG"],
                         LPL2022_Summer_Teams["OMG"],
                         LPL2022_Summer_Teams["FPX"],
                         LPL2022_Summer_Teams["BLG"]))

print("\nLPL 2023 Spring")
LPL2023_Spring_Teams = {
    "AL": ["Zdz", "Xiaohao", "pinz", "Betty", "SwordArt"],
    "BLG": ["Bin", "XUN", "Yagao", "Elk", "ON"],
    "EDG": ["Ale", "Jiejie", "FoFo", "Leave", "Meiko"],
    "FPX": ["Xiaolaohu", "H4cker", "Care", "Lwx", "QiuQiu"],
    "IG": ["YSKM", "gideon", "Dove", "Ahn", "Wink"],
    "JDG": ["369", "Kanavi", "knight", "Ruler", "Missing"],
    "LGD": ["Xiaoxu", "Meteor", "haichao", "Lpc", "Jinjiao"],
    "LNG": ["Zika", "Tarzan", "Scout", "LP", "Hang"],
    "NIP": ["Invincible", "XLB", "Dream", "Photic", "Zhuo"],
    "OMG": ["shanji", "Aki", "Creme", "Able", "ppgod"],
    "RA": ["Cube", "Leyan", "Strive", "Assum", "Southwind"],
    "RNG": ["Breathe", "Wei", "Angel", "GALA", "Ming"],
    "WE": ["Biubiu", "Heng", "Shanks", "Hope", "Iwandy"],
    "TT": ["HOYA", "Beichuan", "ucal", "huanfeng", "yaoyao"],
    "TES": ["Wayward", "Tian", "knight", "JackeyLove", "Mark"],
    "UP": ["Hery", "Ning", "Qing", "Doggo", "Baolan"],
    "WBG": ["TheShy", "Karsa", "Xiaohu", "Light", "Crisp"],
}

for k, v in LPL2023_Spring_Teams.items():
    print(k)
    print(LPL_ELOs[6].get_team_elo(v))
    print(LPL_ELOs[7].get_team_elo(v))

print(
LPL_ELOs[6].LPL_Playoffs(LPL2023_Spring_Teams["JDG"],
                         LPL2023_Spring_Teams["EDG"],
                         LPL2023_Spring_Teams["LNG"],
                         LPL2023_Spring_Teams["WBG"],
                         LPL2023_Spring_Teams["BLG"],
                         LPL2023_Spring_Teams["OMG"],
                         LPL2023_Spring_Teams["TES"],
                         LPL2023_Spring_Teams["TT"],
                         LPL2023_Spring_Teams["RNG"],
                         LPL2023_Spring_Teams["WE"]))
print(
LPL_ELOs[7].LPL_Playoffs(LPL2023_Spring_Teams["JDG"],
                         LPL2023_Spring_Teams["EDG"],
                         LPL2023_Spring_Teams["LNG"],
                         LPL2023_Spring_Teams["WBG"],
                         LPL2023_Spring_Teams["BLG"],
                         LPL2023_Spring_Teams["OMG"],
                         LPL2023_Spring_Teams["TES"],
                         LPL2023_Spring_Teams["TT"],
                         LPL2023_Spring_Teams["RNG"],
                         LPL2023_Spring_Teams["WE"]))

end = datetime.now()
print("Runtime: ", end-start)