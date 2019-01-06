# Import Required Libraries
import numpy as np
import cv2 as cv

def get_Max(matrix):
    largest_num = matrix[0][0]
    for row_idx, row in enumerate(matrix):
        for col_idx, num in enumerate(row):
            if num > largest_num:
                largest_num = num

    return largest_num


def Normalise_Matrix(Matrix):
    row = len(Matrix)
    col = len(Matrix[0])
    MAX_VALUE = get_Max(Matrix)
    for i in range(row):
        for j in range(col):
            Matrix[i][j] = (Matrix[i][j]/MAX_VALUE)*255

    return Matrix


def initialise_matrix(row, col):
    matrix = [[0 for x in range(col)] for y in range(row)]
    return matrix


def elem_wise_multiple(MAT_A, MAT_B, row, col):
    MAT = initialise_matrix(3, 3)
    for i in range(row):
        for j in range(col):
            MAT[i][j] = MAT_A[i][j] * MAT_B[i][j]
    return MAT

def sum_of_elems(MAT):
    value = 0
    row = len(MAT)
    col = len(MAT[0])
    for i in range(row):
        for j in range(col):
            value = value + MAT[i][j]
    
    return value

# funtion to get 3*3 matrix based on its position
def get_3_cross3(matrix, row, col):
    MAT = initialise_matrix(3, 3)
    if row == 0 or col == 0:
        MAT[0][0] = 0
    else:
        MAT[0][0] = matrix[row-1][col-1]

    if row == 0:
        MAT[0][1] = 0
    else:
        MAT[0][1] = matrix[row-1][col]

    if row == 0 or col == len(matrix[0])-1:
        MAT[0][2] = 0
    else:
        MAT[0][2] = matrix[row-1][col+1]

    if col == 0:
        MAT[1][0] = 0
    else:
        MAT[1][0] = matrix[row][col-1]

    MAT[1][1] = matrix[row][col]

    if col == len(matrix[0])-1:
        MAT[1][2] = 0
    else:
        MAT[1][2] = matrix[row][col+1]

    if row == len(matrix)-1 or col == 0:
        MAT[2][0] = 0
    else:
        MAT[2][0] = matrix[row+1][col-1]

    if row == len(matrix)-1:
        MAT[2][1] = 0
    else:
        MAT[2][1] = matrix[row+1][col]

    if row == len(matrix)-1 or col == len(matrix[0])-1:
        MAT[2][2] = 0
    else:
        MAT[2][2] = matrix[row+1][col+1]

    return MAT


# Read the input image in Gray scale format
INPUT_IMAGE = cv.imread('original_imgs/hough.jpg', cv.IMREAD_GRAYSCALE)

# cv.namedWindow("Input Image")
# cv.imshow('Image',inputim)
# cv.waitKey(0)
print(INPUT_IMAGE.shape)


# Initialise the Gx and  Gy matrix
Gx = [[1, 0, -1], 
[2, 0, -2],
 [1, 0, -1]]
Gy = [[1, 2, 1], 
[0, 0, 0],
 [-1, -2, -1]]

rows = len(INPUT_IMAGE)
cols = len(INPUT_IMAGE[0])
INPUTXEDGE = initialise_matrix(rows, cols)
NUM_MAT = initialise_matrix(rows, cols)
INPUTYEDGE = initialise_matrix(rows, cols)

# loop through the matrices and calculate Gx * Input
#for i in range(rows):
#    for j in range(cols):
#        THREE_CROSS_THREE = get_3_cross3(INPUT_IMAGE, i, j)
#        OUTPUT = np.multiply(Gx, THREE_CROSS_THREE)
#        NUM_MAT[i][j] = OUTPUT.sum()

#print(NUM_MAT)

# loop through the matrices and calculate Gx * Input
for i in range(rows):
    for j in range(cols):
        THREE_CROSS_THREE = get_3_cross3(INPUT_IMAGE, i, j)
        OUTPUT = elem_wise_multiple(Gx, THREE_CROSS_THREE, 3, 3)
        INPUTXEDGE[i][j] = sum_of_elems(OUTPUT)

for i in range(rows):
    for j in range(cols):
        THREE_CROSS_THREE = get_3_cross3(INPUT_IMAGE, i, j)
        OUTPUT = elem_wise_multiple(Gy, THREE_CROSS_THREE, 3, 3)
        INPUTYEDGE[i][j] = sum_of_elems(OUTPUT)

OP = initialise_matrix(rows, cols)

for i in range(rows):
    for j in range(cols):
        OP[i][j] = (((INPUTXEDGE[i][j]**2)+(INPUTYEDGE[i][j]**2))**0.5)

INPUTXEDGE = Normalise_Matrix(INPUTXEDGE)
INPUTYEDGE = Normalise_Matrix(INPUTYEDGE)
OP = Normalise_Matrix(OP)

INPUTXEDGE = np.asarray(INPUTXEDGE)
INPUTYEDGE = np.asarray(INPUTYEDGE)
OP = np.asarray(OP)

cv.imwrite( "Edge_along_x.jpg",INPUTXEDGE )
cv.imwrite( "Edge_along_y.jpg",INPUTYEDGE )
cv.imwrite( "output.jpg", OP)
print("Output Image Generated")