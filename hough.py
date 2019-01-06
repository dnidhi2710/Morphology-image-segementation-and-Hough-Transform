import cv2
import numpy as np
from matplotlib import pyplot as plt

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


def detectEdges(image):
    
    INPUT_IMAGE = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

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
    
    print("edge detected")
    return OP

def hough_line(img):
      # Rho and Theta ranges
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas))
  y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos


img = cv2.imread('original_imgs/hough.jpg')
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

output_hsv = hsv_img.copy()
output_hsv[np.where(mask1==0)] = 0

cv2.imwrite("onlyred.jpg",output_hsv)
edges = detectEdges(output_hsv)

cv2.imwrite("edge.jpg",edges)
accumulator, thetas, rhos = hough_line(edges)

idx = np.argmax(accumulator)
rho = rhos[idx / accumulator.shape[1]]
theta = thetas[idx % accumulator.shape[1]]

for rho,theta in zip(thetas,rhos):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite("res1red.jpg",img)
