import pandas as pd
import sqlite3


def create_dataset(tables):
    # Create a dictionary to store the DataFrames
    dfs = {}

    # Iterate over each table
    for table in tables:
        table_name = table[0]
        
        # Read the table data into a DataFrame
        dfs[table_name] = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    print(dfs.keys())

    merged_df = pd.merge(dfs['gas_prices'], dfs['OCED_USA_FOOD_INFLATION'], on='Date', how='inner')

    merged_df.rename({'Price':'gas_price'}, axis=1,inplace=True)

    merged_df = pd.merge(merged_df, dfs['DOW_JONES_REAL'], on='Date', how='inner')

    merged_df.rename({'real-price':'dow_jones_real-price'}, axis=1,inplace=True)

    # merged_df = pd.merge(merged_df, dfs['news_sentiments'], on='Date', how='outer')
    # merged_df['PercentageNegative'] = merged_df['PercentageNegative'].fillna(0)
    # merged_df['PercentageNeutral'] = merged_df['PercentageNeutral'].fillna(0)
    # merged_df['PercentagePositive'] = merged_df['PercentagePositive'].fillna(0)
    # merged_df['TotalSentiments'] = merged_df['TotalSentiments'].fillna(0)


    # check the data types of the columns
    print(merged_df['Date'].dtype)

    # if the 'Date' column is not datetime, convert it to datetime
    if merged_df['Date'].dtype != 'datetime64[ns]':
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    # create a new column 'Year' by extracting the year from the 'Date' column
    merged_df['Year'] = merged_df['Date'].dt.year

    # create a new column 'Month' by extracting the month from the 'Date' column
    merged_df = pd.merge(merged_df, dfs['USA_MEAT_EXPORT_IMPORT'], on='Year', how='inner')

    merged_df = pd.merge(merged_df, dfs['food_production'], on='Year', how='inner')
    merged_df.rename({'lag_1':'cereal_production_lag_1'}, axis=1,inplace=True)
    # check the data types of the columns
    print(dfs['interest_rate']['Date'].dtype)

    # if the 'Date' column is not datetime, convert it to datetime
    if dfs['interest_rate']['Date'].dtype != 'datetime64[ns]':
        dfs['interest_rate']['Date'] = pd.to_datetime(dfs['interest_rate']['Date'])

    merged_df = pd.merge(merged_df, dfs['interest_rate'], on='Date', how='inner')

    return merged_df


if __name__ == "__main__":
    # Connect to the SQLite database
    conn = sqlite3.connect('./food_inflation_analysis.db')
    # Check if connection is successful
    if conn:
        print("Connection is successful")
    else:
        print("Connection failed")
    # Get a list of all tables in the database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    tables = cursor.fetchall()
    print("Tables in the food_inflation_analysis.db database:")
    print(tables)

    merged_df = create_dataset(tables)

    merged_df.to_csv('./data/merged_data.csv', index=False)

    # Close the database connection
    conn.close()
