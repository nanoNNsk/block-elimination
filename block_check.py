import numpy as np

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
    A_inv = np.linalg.inv(A)

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
    while is_row_echelon_form(matrix) == 'f':
        matrix = block_elimination(matrix, block_size)
        block_size += 1
    
    return matrix

# Example matrix from the problem
matrix = np.array([[0, 2, 4],
                   [3, -2, 5]])

# Apply block elimination until the matrix is in row echelon form
result_matrix = apply_block_elimination_until_ref(matrix)
print(result_matrix)

