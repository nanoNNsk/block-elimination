import numpy as np

def create_matrix():
    # Input matrix from user
    matrix_str = input("Input matrix (Separate rows with ; and space between values): ")
    rows = matrix_str.split(';')
    matrix = []
    for row_str in rows:
        row = list(map(float, row_str.split()))
        matrix.append(row)
    
    # Check if all rows have the same length (valid matrix)
    row_length = len(matrix[0])
    if any(len(row) != row_length for row in matrix):
        raise ValueError("Each row must have the same number of columns")
    
    return np.array(matrix)

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        return "Matrices cannot be multiplied"

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))

    return result

def matrix_inverse(A):
    A = np.array(A)
    try:
        inv = np.linalg.inv(A)
        return inv.tolist()
    except np.linalg.LinAlgError:
        return "Matrix is singular and cannot be inverted"
def block_elimination(A):
    # Assume A is a square matrix, partition it into 4 blocks
    n = A.shape[0]
    k = n // 2  # Half point for block separation
    
    # Separate the matrix A into 4 blocks
    A11 = A[:k, :k]  # Top-left block
    A12 = A[:k, k:]  # Top-right block
    A21 = A[k:, :k]  # Bottom-left block
    A22 = A[k:, k:]  # Bottom-right block
    
    # Inverse A11 (must be invertible)
    A11_inv = np.linalg.inv(A11)
    
    # Calculate the Schur complement S = A22 - A21 * A11_inv * A12
    Schur_complement = A22 - np.dot(np.dot(A21, A11_inv), A12)
    
    return A11, A12, A21, Schur_complement

# Create matrix from user input
try:
    matrix = create_matrix()
    matrix2 = create_matrix()
except ValueError as e:
    print(f"Input error: {e}")
    exit()

# Multiply matrix1 by matrix2
resultmultiply = matrix_multiply(matrix, matrix2)

# Inverse of matrix1
inverse = matrix_inverse(matrix)

# Display matrix multiplication result
if type(resultmultiply) == str:
    print(resultmultiply)
else:
    print("Matrix Multiplication Result:")
    for row in resultmultiply:
        print(row)

# Display matrix inverse result
if type(inverse) == str:
    print(inverse)
else:
    print("Matrix Inverse Result:")
    for row in inverse:
        print(row)
A11, A12, A21, Schur_complement = block_elimination(matrix)
print("A11:")
print(A11)
print("A12:")
print(A12)
print("A21:")
print(A21)
print("Schur complement (S):")
print(Schur_complement)