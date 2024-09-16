import numpy as np

def get_matrix():
    rows = int(input("จำนวนแถวของเมทริกซ์: "))
    cols = int(input("จำนวนคอลัมน์ของเมทริกซ์: "))
    
    print("กรอกค่าเมทริกซ์ (คั่นด้วยช่องว่างในแต่ละแถว):")
    A = []
    for i in range(rows):
        row = list(map(float, input(f"แถว {i+1}: ").split()))
        if len(row) != cols:
            print("จำนวนคอลัมน์ไม่ตรงกับค่าที่กรอกมา!")
            return None, None
        A.append(row)
    
    print("กรอกค่าเวกเตอร์ผลลัพธ์ (คั่นด้วยช่องว่าง):")
    b = list(map(float, input().split()))
    
    if len(b) != rows:
        print("จำนวนแถวของเวกเตอร์ผลลัพธ์ไม่ตรงกับค่าที่กรอกมา!")
        return None, None
    
    return np.array(A), np.array(b)

def block_elimination(A, b):
    n, m = A.shape
    
    if n != m:
        return "ไม่สามารถใช้ block elimination กับเมทริกซ์ที่ไม่เป็นสี่เหลี่ยมจัตุรัสได้", [], []

    # แบ่งเมทริกซ์ออกเป็นบล็อกย่อย
    A11 = A[:n//2, :n//2]
    A12 = A[:n//2, n//2:]
    A21 = A[n//2:, :n//2]
    A22 = A[n//2:, n//2:]

    b1 = b[:n//2]
    b2 = b[n//2:]

    # กำจัดบล็อกย่อยด้วยการทำ Schur complement
    try:
        A11_inv = np.linalg.inv(A11)  # หา inverse ของ A11
    except np.linalg.LinAlgError:
        return "ไม่สามารถทำ block elimination ได้ เพราะ A11 ไม่มีการผกผัน", [], []

    # คำนวณ Schur complement
    S = A22 - A21 @ A11_inv @ A12
    b_schur = b2 - A21 @ A11_inv @ b1

    # หาคำตอบของระบบย่อย
    try:
        x2 = np.linalg.solve(S, b_schur)
    except np.linalg.LinAlgError:
        return "ไม่มีคำตอบที่เป็นไปได้", [], []

    x1 = A11_inv @ (b1 - A12 @ x2)

    # รวมคำตอบของทั้งสองบล็อก
    x = np.concatenate([x1, x2])
    
    return x, [], []

def main():
    A, b = get_matrix()
    if A is not None and b is not None:
        result, free_vars, dependent_vars = block_elimination(A, b)
        if isinstance(result, str):  # กรณีที่ไม่มีคำตอบ
            print(result)
        else:
            print("คำตอบของสมการคือ:", result)

if __name__ == "__main__":
    main()
