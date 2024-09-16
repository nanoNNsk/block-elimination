import numpy as np

def get_matrix():
    # รับขนาดเมทริกซ์จากผู้ใช้
    rows = int(input("จำนวนแถวของเมทริกซ์: "))
    cols = int(input("จำนวนคอลัมน์ของเมทริกซ์: "))
    
    # รับค่าของเมทริกซ์ A จากผู้ใช้
    print("กรอกค่าเมทริกซ์ (คั่นด้วยช่องว่างในแต่ละแถว):")
    A = []
    for i in range(rows):
        row = list(map(float, input(f"แถว {i+1}: ").split()))
        if len(row) != cols:
            print("จำนวนคอลัมน์ไม่ตรงกับค่าที่กรอกมา!")
            return None, None
        A.append(row)
    
    # รับค่าของเวกเตอร์ b จากผู้ใช้
    print("กรอกค่าเวกเตอร์ผลลัพธ์ (คั่นด้วยช่องว่าง):")
    b = list(map(float, input().split()))
    
    if len(b) != rows:
        print("จำนวนแถวของเวกเตอร์ผลลัพธ์ไม่ตรงกับค่าที่กรอกมา!")
        return None, None
    
    return np.array(A), np.array(b)

def block_elimination(A, b):
    n, m = A.shape
    free_vars = []
    dependent_vars = []
    
    # ทำ Gaussian elimination และตรวจสอบ free variables
    rank_A = np.linalg.matrix_rank(A)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    rank_augmented = np.linalg.matrix_rank(augmented_matrix)
    
    # ตรวจสอบกรณีไม่มีคำตอบ
    if rank_A < rank_augmented:
        return "ไม่มีคำตอบที่เป็นไปได้ เพราะเมทริกซ์เพิ่มมีอันดับมากกว่าเมทริกซ์ดั้งเดิม"
    
    # Gaussian Elimination with partial pivoting
    for i in range(min(n, m)):
        if A[i, i] == 0:
            # Swap rows to avoid zero pivot
            for k in range(i+1, n):
                if A[k, i] != 0:
                    A[[i, k]] = A[[k, i]]
                    b[[i, k]] = b[[k, i]]
                    break
        
        # Eliminate values below the pivot
        for j in range(i+1, n):
            if A[j, i] != 0:
                factor = A[j, i] / A[i, i]
                A[j, i:] -= factor * A[i, i:]
                b[j] -= factor * b[i]

    # ตรวจจับตัวแปรอิสระ (คอลัมน์ที่ไม่สามารถลดรูปได้สมบูรณ์)
    for i in range(m):
        if i >= n or A[i, i] == 0:
            free_vars.append(f'x{i+1}')
        else:
            dependent_vars.append(f'x{i+1}')

    # กำหนดค่าตัวแปรอิสระเป็น 1
    print(f"ตัวแปรอิสระ: {', '.join(free_vars)} (ถูกแทนด้วยค่า 1)")
    A_solved = A.copy()
    b_solved = b.copy()

    # ตั้งค่าตัวแปรอิสระเป็น 1
    x = np.zeros(m)
    for var in free_vars:
        idx = int(var[1:]) - 1  # แปลงจากชื่อ x1, x2, ... เป็น index
        x[idx] = 1  # แทนค่าตัวแปรอิสระเป็น 1

    # Solve for the dependent variables using back substitution
    result = back_substitution(A_solved, b_solved, x)
    return result, free_vars, dependent_vars

def back_substitution(A, b, x):
    n = len(b)
    # Back substitution to find the solution
    for i in range(n-1, -1, -1):
        if A[i,i] != 0:
            x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    return x

def main():
    A, b = get_matrix()
    if A is not None and b is not None:
        result, free_vars, dependent_vars = block_elimination(A, b)
        if isinstance(result, str):
            print(result)
        else:
            print("คำตอบของสมการคือ:", result)
            if free_vars:
                print(f"ตัวแปรอิสระคือ: {', '.join(free_vars)}")
            if dependent_vars:
                print(f"ตัวแปรที่ขึ้นอยู่คือ: {', '.join(dependent_vars)}")

if __name__ == "__main__":
    main()

