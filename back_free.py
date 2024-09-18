import numpy as np

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

def main():
    # ตัวอย่างการใช้งาน
    A = np.array([[1, -2],
                  [0, 0]])  # รูปแบบ row echelon
    b = np.array([1, 0])
    
    # แก้สมการ
    solve_row_echelon(A, b)

if __name__ == "__main__":
    main()






