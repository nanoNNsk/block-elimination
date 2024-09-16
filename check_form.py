import numpy as np

def is_row_echelon_form(A):
    # จำนวนแถวและคอลัมน์
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

# ตัวอย่างการใช้ฟังก์ชัน
A = np.array([[1, 2, 3, 4], 
              [0, 1, 4, 5], 
              [0, 0, 4, 7]])



result = is_row_echelon_form(A)
print(result)  # จะได้ 't' ถ้าเมทริกซ์อยู่ในรูปแบบขั้นบันได
