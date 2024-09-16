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
    
    A = np.array(A)
    b = np.array(b)
    
    # รวมเมทริกซ์ A และเวกเตอร์ b เป็นเมทริกซ์เดียว
    Ab = np.column_stack((A, b))
    
    return Ab

# เรียกฟังก์ชัน
matrix = get_matrix()
if matrix is not None:
    print("เมทริกซ์รวม:")
    print(matrix)
