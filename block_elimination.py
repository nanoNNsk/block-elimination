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

"""def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if cols_A != rows_B:
        return "Matrices cannot be multiplied"

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))

    return result"""

"""def matrix_inverse(A):
    A = np.array(A)
    try:
        inv = np.linalg.inv(A)
        return inv.tolist()
    except np.linalg.LinAlgError:
        return "Matrix is singular and cannot be inverted"""
def matrix_inverse(A):
    # Convert the input to a numpy array
    A = np.array(A)

    # Number of rows in A
    rows_A = A.shape[0]
    # Number of columns in A
    cols_A = A.shape[1]

    # Check if A is a square matrix
    if rows_A != cols_A:
        return "Matrix is not square"

    # Create an identity matrix of the same size as A
    identity = np.eye(rows_A)

    # Perform Gaussian elimination on A and identity simultaneously
    for i in range(rows_A):
        # Find the row with the largest absolute value in column i
        max_row = np.argmax(abs(A[i:, i])) + i
        # Swap the max row with the current row
        A[[i, max_row]] = A[[max_row, i]]
        identity[[i, max_row]] = identity[[max_row, i]]

        # Divide the current row by the pivot element
        pivot = A[i, i]
        A[i] = A[i] / pivot
        identity[i] = identity[i] / pivot

        # Subtract multiples of the current row from the other rows
        for j in range(rows_A):
            if j != i:
                factor = A[j, i]
                A[j] = A[j] - factor * A[i]
                identity[j] = identity[j] - factor * identity[i]

    # Return the inverse matrix
    return identity

def block_elimination(matrix, block_size):
    """
    Perform block elimination on a matrix of a given block size.
    
    matrix: 2D numpy array (matrix)
    block_size: int, the size of the top-left block A (block_size x block_size)
    
    Returns:
    A new matrix after block elimination.
    """
    # Extract blocks A, B, C, and D from the input matrix
    A = matrix[:block_size, :block_size]               # A is the top-left block
    B = matrix[:block_size, block_size:]               # B is the row vector to the right of A
    C = matrix[block_size:, :block_size]               # C is the column vector below A
    D = matrix[block_size:, block_size:]               # D is the bottom-right block

    # Step 1: Compute the inverse of A
    A_inv = matrix_inverse(A)

    # Step 2: Compute the Schur complement for block D: D - C * A_inv * B
    Schur_complement = D - C @ A_inv @ B

    # Step 3: Construct the resulting matrix
    # A and B remain the same
    # C becomes zeros
    # D is replaced by the Schur complement
    result_matrix = np.block([
        [A, B],
        [np.zeros_like(C), Schur_complement]
    ])

    return result_matrix

def is_row_echelon_form(A):
    """
    Check if a matrix is in row echelon form (REF).
    
    A: 2D numpy array
    
    Returns:
    't' if the matrix is in REF, otherwise 'f'.
    """
    rows, cols = A.shape
    last_pivot_index = -1  # ตำแหน่ง pivot ของแถวก่อนหน้า
    
    for i in range(rows):
        # หา pivot ในแถวปัจจุบัน (ตัวที่ไม่ใช่ศูนย์ตัวแรก)
        pivot_index = -1
        for j in range(cols):
            if A[i, j] != 0:
                pivot_index = j
                break
        
        if pivot_index == -1:
            # แถวนี้เป็นแถวศูนย์ทั้งหมด ตรวจสอบว่าแถวถัดไปก็ต้องเป็นศูนย์ทั้งหมดด้วย
            continue
        
        if pivot_index <= last_pivot_index:
            # pivot ของแถวนี้ต้องอยู่ทางขวาของ pivot ของแถวก่อนหน้า
            return 'f'
        
        last_pivot_index = pivot_index
    
    return 't'

def apply_block_elimination_until_ref(matrix):
    """
    Apply block elimination repeatedly, increasing block size, until the matrix is in row echelon form.
    
    matrix: 2D numpy array (matrix)
    
    Returns:
    The matrix in row echelon form after performing block elimination.
    """
    block_size = 1
    step = 1
    
    while is_row_echelon_form(matrix) == 'f':
        print(f"Step {step}: Matrix is not in row echelon form. Applying block elimination with block size {block_size}...")
        matrix = block_elimination(matrix, block_size)
        print("Matrix after block elimination:")
        print(matrix)
        block_size += 1
        step += 1
    
    print("Matrix is now in row echelon form:")
    return matrix
"""
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
    A11_inv = matrix_inverse(A11)
    
    # Calculate the Schur complement S = A22 - A21 * A11_inv * A12
    Schur_complement = A22 - np.dot(np.dot(A21, A11_inv), A12)
    
    return A11, A12, A21, Schur_complement
    """

# Create matrix from user input
try:
    matrix = create_matrix()
    #matrix2 = create_matrix()
except ValueError as e:
    print(f"Input error: {e}")
    exit()

# Multiply matrix1 by matrix2
#resultmultiply = matrix_multiply(matrix, matrix2)

# Inverse of matrix1
inverse = matrix_inverse(matrix)

# Display matrix multiplication result
"""if type(resultmultiply) == str:
    print(resultmultiply)
else:
    print("Matrix Multiplication Result:")
    for row in resultmultiply:
        print(row)"""

# Display matrix inverse result
if type(inverse) == str:
    print(inverse)
else:
    print("Matrix Inverse Result:")
    for row in inverse:
        print(row)
try:
    result_matrix = apply_block_elimination_until_ref(matrix)
except ValueError as e:
    print(f"Input error: {e}")
    exit()
print(result_matrix)
"""
A11, A12, A21, Schur_complement = block_elimination(matrix)
print("A11:")
print(A11)
print("A12:")
print(A12)
print("A21:")
print(A21)
print("Schur complement (S):")
print(Schur_complement)
"""