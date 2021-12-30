# import required libraries
import pandas as pd
import openpyxl

# load the cmeg training data set:
df1 = pd.read_excel("cmeg_df_case_competition_scrambled_train.xlsx")


# read_excel reads the first sheet within the workbook by default
# We can also read the second sheet with the data key
dk1 = pd.read_excel("cmeg_df_case_competition_scrambled_train.xlsx", sheet_name=1)

# load the general_industries training data set:
df2 = pd.read_excel("general_industries_df_case_competition_scrambled_train.xlsx")

# read the second sheet with the data key
dk2 = pd.read_excel("general_industries_df_case_competition_scrambled_train.xlsx", sheet_name=1)

# save in CSV format (without index column)
df1.to_csv("cmeg_df_case_competition_scrambled_train.csv", index=False)
dk1.to_csv("cmeg_df_case_competition_scrambled_train_datakey.csv", index=False)
df2.to_csv("general_industries_df_case_competition_scrambled_train.csv", index=False)
dk2.to_csv("general_industries_df_case_competition_scrambled_train_datakey.csv", index=False)