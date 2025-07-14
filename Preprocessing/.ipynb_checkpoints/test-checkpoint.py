import pandas as pd

data=pd.read_csv("/Raw_Data/movies.csv")
data_df=pd.DataFrame(data)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Prevent line wrapping

print(data_df['overview'])

