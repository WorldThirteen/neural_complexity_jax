import numpy as np
import re

def parse_file_to_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract matrix dimensions from the file
    matrix_size = re.search(r'Matrix size = (\d+) rows x (\d+) columns', lines[0])
    num_rows = int(matrix_size.group(1))
    num_cols = int(matrix_size.group(2))

    matrix = np.zeros((num_rows, num_cols))

    # Iterate through lines to extract matrix values
    for line in lines[4:]:  # Skip the first 4 lines
        match = re.match(r'set row=(\d+), col=(\d+) = ([\d\.\-eE]+)', line)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            value = float(match.group(3))
            matrix[row, col] = value

    return matrix

if __name__ == "__main__":
    file_path = 'polyworld.txt'
    matrix = parse_file_to_matrix(file_path)
    np.set_printoptions(precision=6, suppress=True)

    np.savetxt('output_matrix.csv', matrix, delimiter=',', fmt='%.9f')
