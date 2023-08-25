# LoL-ELOs-Project
## Freiwald Scholars Project, Summer 2023
### By Avery Wang
This is a collection of all of the code and CSV files used in my project for the Freiwald Scholars Program in Summer 2023. I will continue to work on this in the future, so any ideas for features to add are appreciated!

## CSV Files and Reading Data
The files with names such as [[Year]_LoL_esports_match_data_from_OraclesElixir.csv](../main/2019_LoL_esports_match_data_from_OraclesElixir.csv) are the raw files downloaded from [Oracle's Elixir](https://oracleselixir.com/tools/downloads). They were downloaded on June 9, 2023, but in a future version, these may be redownloaded daily so that match data remains up-to-date. These files are used in [Read_data.py](../main/Read_data.py) to create the CSV files such as [LCK.csv](../main/LCK.csv) and [LCK0.csv](../main/LCK.csv). These files should not be touched!

## Ideal K-Factor
The file [IdealK.py](../main/IdealK.py) is used to calculate the best performing K-Factor in each of the four leagues between Spring 2019 and Summer 2023, including regular season and playoffs.

## Playoff Predictions
The file [Predictions.ipynb](../main/Predictions.ipynb) is used to make predictions for each playoff bracket between Spring 2019 and Summer 2023.

## Results
Ideal K-Factor and Playoff Predictions results can be found in [Results.xlsx](../main/Results.xlsx).
