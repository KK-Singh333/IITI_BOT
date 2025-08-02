import pandas as pd
import re
import os

input_csv_file = 'output_data.csv'

# Change this to the name of the new CSV file you want to create
output_csv_file = 'cleaned_data.csv'

# Change this to the name of the column you want to clean
column_to_clean = 'body_text'
# ---------------------

def remove_devanagari(text):
    """
    Removes characters from the Devanagari Unicode block.
    This includes Hindi, Marathi, Nepali, and other languages.
    """
    if isinstance(text, str):
        # The regular expression pattern [\u0900-\u097F] matches all
        # characters in the Devanagari Unicode range.
        return re.sub(r'[\u0900-\u097F]+', '', text)
    return text

def clean_data(input_file, output_file, column_name):
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found.")
        return

    print(f"Reading data from '{input_file}'...")
    try:
        df = pd.read_csv(input_file)
        df = df[df[column_name].notna() & (df[column_name].str.strip() != '')]
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    df[column_name] = df[column_name].apply(remove_devanagari)

    print(f"Saving cleaned data to '{output_file}'...")
    df.to_csv(output_file, index=False)

    print(f"✅ Process complete! Cleaned data is saved as '{output_file}'.")

# Run the function with the configured file names
clean_data(input_csv_file, output_csv_file, column_to_clean)