from pathlib import Path
import pandas as pd

def remove_column_from_csv(csv_file_path, column_name, output_file_path=None):
    """
    Remove a specific column from a CSV file and save the result to a new file or overwrite the original file.

    Args:
    csv_file_path (Path): Path to the input CSV file.
    column_name (str): Name of the column to be removed.
    output_file_path (Path, optional): Path to save the updated CSV file. If not provided, overwrites the original file.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)
    
    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        # Remove the specified column
        df = df.drop(columns=[column_name])
        print(f"Column '{column_name}' removed.")
    else:
        print(f"Column '{column_name}' not found in the CSV file.")
    
    # Save the updated DataFrame back to a CSV file
    if output_file_path:
        df.to_csv(output_file_path, index=False)
        print(f"Updated CSV file saved to {output_file_path}")
    else:
        df.to_csv(csv_file_path, index=False)
        print(f"Updated CSV file saved to {csv_file_path}")

csv_file = Path(r'results\runs_history.csv')  
output_file = Path(r'results\runs_history_updated.csv')  

remove_column_from_csv(csv_file_path=csv_file, column_name='num_layers', output_file_path=output_file)
