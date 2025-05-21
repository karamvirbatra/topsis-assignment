import pandas as pd
import numpy as np
import sys

def validate_inputs(input_file, weights, impacts):
    try:
        data = pd.read_csv(input_file)
        if data.shape[1] < 3:
            raise ValueError("Input file must contain at least three columns.")
        
        for col in data.columns[1:]:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError("All columns except the first must contain numeric values.")
        
        weights = list(map(float, weights.split(',')))
        impacts = impacts.split(',')
        
        if len(weights) != len(impacts) or len(weights) != data.shape[1] - 1:
            raise ValueError("Number of weights, impacts, and numeric columns must match.")
        
        if not all(i in ['+', '-'] for i in impacts):
            raise ValueError("Impacts must be '+' or '-'.")
        
        return data, weights, impacts
    except FileNotFoundError:
        sys.exit(f"Error: File '{input_file}' not found.")
    except Exception as e:
        sys.exit(f"Error: {str(e)}")

def topsis(data, weights, impacts):
    # Normalize the decision matrix
    norm_data = data.iloc[:, 1:].apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=0)
    
    # Apply weights
    weighted_data = norm_data * weights
    
    # Determine ideal best and worst
    ideal_best = []
    ideal_worst = []
    for i, impact in enumerate(impacts):
        if impact == '+':
            ideal_best.append(weighted_data.iloc[:, i].max())
            ideal_worst.append(weighted_data.iloc[:, i].min())
        else:
            ideal_best.append(weighted_data.iloc[:, i].min())
            ideal_worst.append(weighted_data.iloc[:, i].max())
    
    # Calculate distances
    distances_best = np.sqrt(((weighted_data - ideal_best) ** 2).sum(axis=1))
    distances_worst = np.sqrt(((weighted_data - ideal_worst) ** 2).sum(axis=1))
    
    # Calculate TOPSIS scores
    scores = distances_worst / (distances_best + distances_worst)
    
    # Rank the scores
    data['Topsis Score'] = scores
    data['Rank'] = scores.rank(ascending=False).astype(int)
    return data

def main():
    if len(sys.argv) != 5:
        sys.exit("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]
    
    # Validate inputs
    data, weights, impacts = validate_inputs(input_file, weights, impacts)
    
    # Perform TOPSIS
    result = topsis(data, weights, impacts)
    
    # Save results
    result.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")

if __name__ == "__main__":
    # Automatically save uploaded Excel file as CSV for input
    input_excel = 'data.xlsx'  # Name of the uploaded Excel file
    csv_file = 'data.csv'      # Converted CSV file
    try:
        data = pd.read_excel(input_excel)
        data.to_csv(csv_file, index=False)
        print(f"Converted '{input_excel}' to '{csv_file}'.")
    except Exception as e:
        sys.exit(f"Error converting Excel to CSV: {e}")

    # Run the main function
    main()
