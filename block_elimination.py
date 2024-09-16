def create_matrix():
    #input matrix
    matrix_str = input("input matrix(Separate rows with ; and please leave a space when entering each value): ")
    rows = matrix_str.split(';')  #split row
    matrix = []
    for row_str in rows:
        row = list(map(float, row_str.split()))  #change to number float
        matrix.append(row)
    return matrix

#create matrix
matrix = create_matrix()
#check matrix
for row in matrix:
    print(row)