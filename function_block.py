import numpy as np

def block_elimination(matrix):
    """
    Perform block elimination on a 4x4 matrix, dividing it into blocks A, B, C, and D.
    Then, compute the Schur complement and set C to zeros.
    
    matrix: 2D numpy array (4x4 matrix)
    
    Returns:
    A new matrix after block elimination.
    """
    # Extract blocks A, B, C, and D from the input matrix
    A = matrix[0, 0]               # A is a scalar (the top-left element)
    B = matrix[0, 1:]              # B is the row vector to the right of A
    C = matrix[1:, 0]              # C is the column vector below A
    D = matrix[1:, 1:]             # D is the bottom-right block (3x3 matrix)

    # Step 1: Compute the inverse of A (since A is a scalar, its inverse is 1/A)
    A_inv = 1 / A

    # Step 2: Compute the Schur complement for block D: D - C * A_inv * B
    Schur_complement = D - np.outer(C, B) * A_inv

    # Step 3: Construct the resulting matrix
    # A remains the same
    # B remains the same
    # C becomes zeros
    # D is replaced by the Schur complement
    result_matrix = np.block([
        [np.array([[A]]), B.reshape(1, -1)],
        [np.zeros_like(C).reshape(-1, 1), Schur_complement]
    ])

    return result_matrix

# Example matrix (from the problem)
matrix = np.array([[2, 1, 0],
                   [-2, 0, 1],
                   [8, 5, 3]])

# Perform block elimination
result = block_elimination(matrix)
print("Resultant matrix after block elimination:")
print(result)


