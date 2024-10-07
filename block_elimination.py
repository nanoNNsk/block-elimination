import numpy as np

def create_matrix():
    # Input matrix from user
    matrix_str = input("Input matrix (Separate rows with ; and space between values): ")
    rows = matrix_str.split(';')
    matrix = []
    for row_str in rows:
        try:
            row = list(map(float, row_str.split()))  # Ensure all values are converted to floats
        except ValueError:
            raise ValueError("Input contains non-numeric values. Please enter only numeric values.")
        matrix.append(row)
    if matrix[0][0] == 0:
        matrix[1],matrix[0] = matrix[0],matrix[1]
        print("swap row 1 and 2 because first number of row one is 0")
        
    
    # Check if all rows have the same length (valid matrix)
    row_length = len(matrix[0])
    if any(len(row) != row_length for row in matrix):
        raise ValueError("Each row must have the same number of columns")
    
    return np.array(matrix)

def generate_random_matrix(rows, cols):
    """Generates a random matrix of the specified dimensions.

    Args:
        rows: The number of rows in the matrix.
        cols: The number of columns in the matrix.

    Returns:
        A NumPy array representing the random matrix.   

    """

    return np.random.rand(rows, cols)

np.seterr(divide='ignore', invalid='ignore') #for ignore RuntimeWarning

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
    # Ensure matrix is of type float64
    matrix = np.array(matrix, dtype=np.float64)

    # Extract blocks A, B, C, and D from the input matrix
    A = matrix[:block_size, :block_size].astype(np.float64)  # A is the top-left block
    B = matrix[:block_size, block_size:].astype(np.float64)  # B is the row vector to the right of A
    C = matrix[block_size:, :block_size].astype(np.float64)  # C is the column vector below A
    D = matrix[block_size:, block_size:].astype(np.float64)  # D is the bottom-right block

    # Step 1: Check if A is invertible (non-zero pivot), skip the block if singular
    try:
        A_inv = matrix_inverse(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix A is singular, cannot proceed with block elimination.")

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
    Apply block elimination repeatedly, increasing block size, until the matrix is in row echelon form
    or no further elimination is possible.
    
    matrix: 2D numpy array (matrix)
    
    Returns:
    The matrix in row echelon form after performing block elimination or after handling a singular block.
    """
    block_size = 1
    step = 1
    max_block_size = min(matrix.shape)  # Maximum block size is limited by the matrix dimensions
    prev_matrix = None  # To track changes between iterations
    
    while is_row_echelon_form(matrix) == 'f':
        #print(f"Step {step}: Matrix is not in row echelon form. Applying block elimination with block size {block_size}...")
        
        # Store the current matrix for comparison after elimination
        prev_matrix = matrix.copy()
        
        # Apply block elimination
        matrix = block_elimination(matrix, block_size)
        
        #print("Matrix after block elimination:")
        #print(matrix)
        
        # Check if the matrix has changed since the last step (to avoid infinite loops)
        if np.allclose(matrix, prev_matrix):
            print("Matrix stopped changing. Terminating the process.")
            break
        
        # Increase the block size
        block_size += 1
        step += 1
        
        # Ensure we don't exceed the matrix dimensions
        if block_size > max_block_size:
            #print("Block size exceeds matrix dimensions. Terminating the process.")
            break
    
    #print("Matrix is now in row echelon form or block elimination cannot proceed further.")
    return matrix


def back_substitution(A, b, x):
    """
    Perform back substitution to solve the system of equations.
    
    Parameters:
    A : 2D numpy array (row echelon matrix)
    b : 1D numpy array (result vector)
    x : 1D numpy array (solution vector with free variables set to 1)
    
    Returns:
    A numpy array of the solution.
    """
    rows, cols = A.shape
    
    # Back substitution for pivot variables
    for i in range(rows - 1, -1, -1):
        row = A[i]
        b_val = b[i]
        for j in range(cols):
            if row[j] != 0:
                x[j] = (b_val - np.dot(row[j+1:], x[j+1:])) / row[j]
                break
    
    return x

def is_inconsistent(A, b):
    """
    Check if the system is inconsistent (no solution).
    
    Parameters:
    A : 2D numpy array (row echelon matrix)
    b : 1D numpy array (result vector)
    
    Returns:
    True if the system is inconsistent, False otherwise.
    """
    rows, cols = A.shape
    for i in range(rows):
        if np.all(A[i, :] == 0) and b[i] != 0:
            return True
    return False

def find_free_and_dependent_variables(A):
    """
    Find the free and dependent variables in a matrix in row echelon form (REF).
    
    Parameters:
    A : 2D numpy array (row echelon matrix)
    
    Returns:
    free_vars : List of free variable indices (list of integers)
    dependent_vars : List of dependent variable indices (list of integers)
    """
    rows, cols = A.shape
    pivot_cols = []
    free_vars = []
    
    # หาตำแหน่ง pivot ในแต่ละแถว
    for i in range(rows):
        for j in range(cols):
            if A[i, j] != 0:
                pivot_cols.append(j)
                break
    
    # ตัวแปรอิสระคือคอลัมน์ที่ไม่มี pivot
    for j in range(cols):
        if j not in pivot_cols:
            free_vars.append(j)
    
    dependent_vars = [j for j in range(cols) if j not in free_vars]
    
    return free_vars, dependent_vars

def solve_row_echelon(A, b):
    """
    Solve a system of linear equations with a row echelon matrix.
    
    Parameters:
    A : 2D numpy array (row echelon matrix)
    b : 1D numpy array (result vector)
    
    Returns:
    The solution vector with free variables substituted as 1.
    """
    # ตรวจสอบว่าระบบสมการไม่มีคำตอบหรือไม่
    if is_inconsistent(A, b):
        print("ระบบสมการนี้ไม่มีคำตอบ (Inconsistent system)")
        return None
    
    rows, cols = A.shape
    x = np.zeros(cols)
    
    # หาตัวแปรอิสระและตัวแปรที่ขึ้นอยู่
    free_vars, dependent_vars = find_free_and_dependent_variables(A)
    
    # กำหนดค่าตัวแปรอิสระเป็น 1
    for var in free_vars:
        x[var] = 1
    
    # แก้สมการสำหรับตัวแปรที่ขึ้นอยู่โดยใช้ back substitution
    x = back_substitution(A, b, x)
    
    # สร้างชื่อของตัวแปร
    var_names = [f"x{i+1}" for i in range(cols)]

    # แสดงผลตัวแปรอิสระ
    if free_vars:
        free_vars_names = [var_names[i] for i in free_vars]
        print(f"ตัวแปรอิสระ: {', '.join(free_vars_names)} (ถูกแทนด้วยค่า 1)")
    else:
        print("ไม่มีตัวแปรอิสระ")
    
    # แสดงผลคำตอบของสมการ
    print(f"คำตอบของสมการคือ: {x}")
    return x, free_vars, dependent_vars


# Create matrix from user input
try:
    matrix = create_matrix()
    #matrix = generate_random_matrix(15, 15)
    #matrix2 = create_matrix()
except ValueError as e:
    print(f"Input error: {e}")
    exit()

#print("before block elimination")
#print(matrix)

try:
    ref_matrix = apply_block_elimination_until_ref(matrix)
except ValueError as e:
    print(f"Input error: {e}")
    exit()

#check row echelon form again
if not is_row_echelon_form(ref_matrix):
    print("Error: The matrix is not in row echelon form.")
    exit()
# Extract last column as b and remove it from matrix A
b = ref_matrix[:, -1]
ref_matrix = ref_matrix[:, :-1]  # Remove the last column to get the coefficient matrix A

# Solve the system using back substitution
try:
    solution = solve_row_echelon(ref_matrix, b)
    #print("Solution to the system is:")
    #print(solution)
except ValueError as e:
    print(f"Error during back substitution: {e}")
