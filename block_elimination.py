def create_matrix():
    #input matrix
    matrix_str = input("input matrix(Separate rows with ; and please leave a space when entering each value): ")
    rows = matrix_str.split(';')  #split row
    matrix = []
    for row_str in rows:
        row = list(map(float, row_str.split()))  #change to number float
        matrix.append(row)
    return matrix

def matrix_multiply(A, B):
    # Number of rows in A
    rows_A = len(A)
    # Number of columns in A (or rows in B)
    cols_A = len(A[0])
    # Number of columns in B
    cols_B = len(B[0])

    # Check if multiplication is possible
    if cols_A != len(B):
        return "Matrices cannot be multiplied"

    # Create a result matrix filled with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Multiply matrices
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

def matrix_inverse(A):
    # Number of rows in A
    rows_A = len(A)
    # Number of columns in A
    cols_A = len(A[0])

    # Check if A is a square matrix
    if rows_A != cols_A:
        return "Matrix is not square"

    # Create an identity matrix of the same size as A
    identity = [[float(i == j) for i in range(rows_A)] for j in range(cols_A)]

    # Perform Gaussian elimination on A and identity simultaneously
    for i in range(rows_A):
        # Find the row with the largest absolute value in column i
        max_row = max(range(i, rows_A), key=lambda r: abs(A[r][i]))
        # Swap the max row with the current row
        A[i], A[max_row] = A[max_row], A[i]
        identity[i], identity[max_row] = identity[max_row], identity[i]

        # Divide the current row by the pivot element
        pivot = A[i][i]
        A[i] = [x / pivot for x in A[i]]
        identity[i] = [x / pivot for x in identity[i]]

        # Subtract multiples of the current row from the other rows
        for j in range(rows_A):
            if j != i:
                factor = A[j][i]
                A[j] = [A[j][k] - factor * A[i][k] for k in range(cols_A)]
                identity[j] = [identity[j][k] - factor * identity[i][k] for k in range(cols_A)]

    # Return the inverse matrix
    return identity

#create matrix
matrix = create_matrix()
matrix2 = create_matrix()
resultmultiply = matrix_multiply(matrix,matrix2)
inverse = matrix_inverse(matrix)
#check matrix multiply
if type(resultmultiply) == str:
    print(resultmultiply)
else:
    for row in resultmultiply:
        print(row)
#check matrix inverse
if type(inverse) == str:
    print(inverse)
else:
    for row in inverse:
        print(row)