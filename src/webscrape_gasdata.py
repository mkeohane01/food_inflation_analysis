import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page
url = "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=emm_epmr_pte_nus_dpg&f=m"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table in the HTML
    table = soup.find("table", {"class": "FloatTitle"})

    # Extract the rows from the table
    rows = table.find_all('tr')

    # Initialize an empty list to store the data
    data = []

    # Loop over the rows and extract the data
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])  # Get rid of empty values

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    # Rename the columns to months
    df.columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct','Nov','Dec']

    # Set the index to the Year column
    df = df.set_index('Year')

    # Convert the data to numeric values
    df = df.apply(pd.to_numeric, errors='coerce')

    # remove rows that are all NaN
    df = df.dropna(how='all')

    # edit 1993 data to move nans to first 3 months instead of last 3 columns
    df.loc['1990'] = df.loc['1990'].shift(7)

    # Display the DataFrame
    print(df)

    # save the dataframe to a csv file
    df.to_csv('data/gas_prices.csv')
else:
    print("Failed to retrieve the page. Status code:", response.status_code)
