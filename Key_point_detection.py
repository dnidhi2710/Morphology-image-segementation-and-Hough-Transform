# This python program is to get an input image and find out the key points in the given image
# using SIFT Detection Algorithm
import numpy as np
import cv2 as cv
import math as math

#This function generates a matrix with given row,col size filled with zeros
def initialise_matrix(row, col):
    matrix = [[0 for x in range(col)] for y in range(row)]
    return matrix

#this function returns the max value for the given matrix
def get_Max(matrix):
    largest_num = matrix[0][0]
    for row_idx, row in enumerate(matrix):
        for col_idx, num in enumerate(row):
            if num > largest_num:
                largest_num = num

    return largest_num

#This function normalises the image matrix to 0-255 scale
def Normalise_Matrix(Matrix):
    row = len(Matrix)
    col = len(Matrix[0])
    MAX_VALUE = get_Max(Matrix)
    for i in range(row):
        for j in range(col):
            Matrix[i][j] = (Matrix[i][j]/MAX_VALUE)*255
    return Matrix

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

#this function pads 0 for the edge rows and cols
def generatePatchMatrix(matrix, row, col):
    PATCH_MAT = initialise_matrix(5, 5)
    row_i = row - 2
    for i in range(5):
        col_i = col - 2
        for j in range(5):
            if row_i < 0 or row_i > len(matrix)-1 or col_i < 0 or col_i > len(matrix[0])-1:
                PATCH_MAT[i][j] = 0
            else:
                PATCH_MAT[i][j] = matrix[row_i][col_i]
            col_i = col_i + 1

        if i > 4:
            row_i = row - 2
        else:
            row_i = row_i + 1
    return PATCH_MAT

# this function normalised the gaussian kernel
def GaussianNormalisation(matrix):
    row = len(matrix)
    col = len(matrix[0])
    total = 0
    for i in range(row):
        for j in range(col):
            total = total + matrix[i][j]

    for i in range(row):
        for j in range(col):
            matrix[i][j] = matrix[i][j]/total
    return matrix

# this function generates the gaussian kernel for the given Sigma and weight
def generateGaussianKernel(weight, sigma):
    matrix = initialise_matrix(weight, weight)
    row_i = -2
    for i in range(weight):
        col_i = -2
        for j in range(weight):
            power_elem = -((row_i**2)+(col_i**2))/(2*(sigma**2))
            matrix[i][j] = (1/(2*math.pi*(sigma**2)))*math.exp(power_elem)
            col_i = col_i + 1

        if i == 4:
            row_i = -2
        else:
            row_i = row_i + 1

    matrix = GaussianNormalisation(matrix)
    return matrix

# this function returns the convoluted matrix
def computeConvolutedMat(matrix):
    row = len(matrix)
    col = len(matrix[0])
    row_i = row - 1
    col_i = col - 1
    OUTPUT = initialise_matrix(row, col)
    for i in range(row):
        col_i = col - 1
        for j in range(col):
            OUTPUT[i][j] = matrix[row_i][col_i]
            col_i = col_i - 1
        if i == 457:
            row_i = row - 1
        else:
            row_i = row_i - 1

    return OUTPUT

# this function does element wise multiplication of the given 2 matrices
def elem_wise_operation(kernel, pos):
    op = initialise_matrix(5, 5)
    for i in range(5):
        for j in range(5):
            op[i][j] = kernel[i][j]*pos[i][j]
    return op

# this function returns the sum of all the values in a matrix
def sum_of_elems(MAT):
    value = 0
    row = len(MAT)
    col = len(MAT[0])
    for i in range(row):
        for j in range(col):
            value = value + MAT[i][j]

    return value

# this is a main function which applies gaussian filter to each n every 
# pixel in a given input image
def AddGaussianFilterToImage(sigma, image):
    GaussianKernel = generateGaussianKernel(5, sigma)
    op_mat = initialise_matrix(len(image), len(image[0]))
    for i in range(len(image)):
        for j in range(len(image[0])):
            pos_mat = generatePatchMatrix(image, i, j)
          #  pos_con_mat = computeConvolutedMat(pos_mat)
            computed_mat = elem_wise_operation(pos_mat, GaussianKernel)
            op_mat[i][j] = sum_of_elems(computed_mat)

    return op_mat

# this function scales down the given image matrix to half
def ScaleDownImage(Mat):
    row = len(Mat)
    col = len(Mat[0])
    row_i = row // 2
    col_i = col // 2
    op = initialise_matrix(row_i, col_i)
    for i in range(row_i):
        for j in range(col_i):
            op[i][j] = Mat[2*i][2*j]

    return op

# this function calculates the DoG of given 2 Matrices
# Gaussian 1 - Gaussian 2.. Element wise subtraction
def calculate_DOG(Mat1, Mat2):
    row = len(Mat1)
    col = len(Mat1[0])
    output = initialise_matrix(row,col)
    for i in range(row):
        for j in range(col):
            output[i][j] = Mat1[i][j] - Mat2[i][j]
    
    return output

#this function get three matrices and check whether the value is maximum of minimum
def getKeyPoint(Mat1,Mat2,Mat3,Value):
    row = len(Mat1)
    col = len(Mat1[0])
    complete_list = []
    for i in range(row):
        for j in range(col):
            complete_list.append(Mat1[i][j])
            complete_list.append(Mat2[i][j])
            complete_list.append(Mat3[i][j])
    
    max_value = max(complete_list)
    min_value = min(complete_list)

    if Value == max_value or Value == min_value:
        return Value
    else:
        return 0

#this function returns and output matrix with only the maxima and minima values
def calculate_keypoints(Mat1,Mat2,Mat3):
    row = len(Mat1)
    col = len(Mat1[0])
    op = initialise_matrix(row,col)
    for i in range(row):
        for j in range(col):
            Pix1 =  get_3_cross3(Mat1,i,j)
            Pix2 =  get_3_cross3(Mat2,i,j)
            Pix3 =  get_3_cross3(Mat3,i,j)
            op[i][j] = getKeyPoint(Pix1,Pix2,Pix3,Mat2[i][j])

    return op

def mark_keypoints(Key1,Key2,Color_image):
    row = len(Color_image)
    col = len(Color_image[0])
    for i in range(row):
        for j in range(col):
            if Key1[i][j] > 3.0 or Key2[i][j] > 3.0:
                Color_image[i][j] = 255
    return Color_image

def octave1_gaussian(INPUT_IMAGE):
    # OCTAVE 1 Code starts here
    # Computing the Matrix for the given original image for various sigma values
    # Sigma values = 1/root2,1, root2,2,2root2
    OCT1GAUS1 = AddGaussianFilterToImage(0.707, INPUT_IMAGE)
    OCT1GAUS2 = AddGaussianFilterToImage(1, INPUT_IMAGE)
    OCT1GAUS3 = AddGaussianFilterToImage(1.414, INPUT_IMAGE)
    OCT1GAUS4 = AddGaussianFilterToImage(2, INPUT_IMAGE)
    OCT1GAUS5 = AddGaussianFilterToImage(2.828, INPUT_IMAGE)
    print("Octave 1 Gaussian Completed")
    cv.imwrite( "octave1_Gaus1.jpg", np.asarray(Normalise_Matrix(OCT1GAUS1)))
    cv.imwrite( "octave1_Gaus2.jpg", np.asarray(Normalise_Matrix(OCT1GAUS2)))
    cv.imwrite( "octave1_Gaus3.jpg", np.asarray(Normalise_Matrix(OCT1GAUS3)))
    cv.imwrite( "octave1_Gaus4.jpg", np.asarray(Normalise_Matrix(OCT1GAUS4)))
    cv.imwrite( "octave1_Gaus5.jpg", np.asarray(Normalise_Matrix(OCT1GAUS5)))
    #These function calls calculate the difference of Gaussian
    #Store it in the respective Identifiers
    OCT1DOG1 = calculate_DOG(OCT1GAUS1,OCT1GAUS2)
    OCT1DOG2 = calculate_DOG(OCT1GAUS2,OCT1GAUS3)
    OCT1DOG3 = calculate_DOG(OCT1GAUS3,OCT1GAUS4)
    OCT1DOG4 = calculate_DOG(OCT1GAUS4,OCT1GAUS5)
    print("Octave 1 DoG Completed")
    cv.imwrite( "octave1_dog1.jpg", np.asarray(Normalise_Matrix(OCT1DOG1)))
    cv.imwrite( "octave1_dog2.jpg", np.asarray(Normalise_Matrix(OCT1DOG2)))
    cv.imwrite( "octave1_dog3.jpg", np.asarray(Normalise_Matrix(OCT1DOG3)))
    cv.imwrite( "octave1_dog4.jpg", np.asarray(Normalise_Matrix(OCT1DOG4)))

    #This part of code Locates the maxima and minima to find the keypoints
    OCT1Key1 = calculate_keypoints(OCT1DOG1,OCT1DOG2,OCT1DOG3)
    OCT1Key2 = calculate_keypoints(OCT1DOG2,OCT1DOG3,OCT1DOG4)
    #cv.imshow('octave1_Key1', np.asarray(OCT1Key1))
    cv.imwrite( "octave1_key1.jpg", np.asarray(Normalise_Matrix(OCT1Key1)))
    cv.imwrite( "octave1_key2.jpg", np.asarray(Normalise_Matrix(OCT1Key2)))
    print("Octave 1 Keypoint images Generated")
    #cv.waitKey(0)
    
    #this part of code points the generated key point value in original image
    OCT1FINAL = mark_keypoints(OCT1Key1,OCT1Key2,INPUT_IMAGE)
    ORGFINAL = mark_keypoints(OCT1Key1,OCT1Key2,Color_image)
    cv.imwrite("octave1_final.jpg",np.asarray(OCT1FINAL))
    cv.imwrite("original_key_final.jpg",np.asarray(ORGFINAL))
    print("Octave 1 final Image generated")

def octave2_gaussian(second_image):
    # OCTAVE 2 Code starts here
    # Computing the Matrix for the given original image for various sigma values
    # Sigma values = 1.414,2,2.828,4,5.657
    OCT2GAUS1 = AddGaussianFilterToImage(1.414, second_image)
    OCT2GAUS2 = AddGaussianFilterToImage(2, second_image)
    OCT2GAUS3 = AddGaussianFilterToImage(2.828, second_image)
    OCT2GAUS4 = AddGaussianFilterToImage(4, second_image)
    OCT2GAUS5 = AddGaussianFilterToImage(5.657, second_image)
    print("Octave 2 Gaussian Completed")
    cv.imwrite( "octave2_Gaus1.jpg", np.asarray(Normalise_Matrix(OCT2GAUS1)))
    cv.imwrite( "octave2_Gaus2.jpg", np.asarray(Normalise_Matrix(OCT2GAUS2)))
    cv.imwrite( "octave2_Gaus3.jpg", np.asarray(Normalise_Matrix(OCT2GAUS3)))
    cv.imwrite( "octave2_Gaus4.jpg", np.asarray(Normalise_Matrix(OCT2GAUS4)))
    cv.imwrite( "octave2_Gaus5.jpg", np.asarray(Normalise_Matrix(OCT2GAUS5)))

    #These function calls calculate the difference of Gaussian
    #Store it in the respective Identifiers
    OCT2DOG1 = calculate_DOG(OCT2GAUS1,OCT2GAUS2)
    OCT2DOG2 = calculate_DOG(OCT2GAUS2,OCT2GAUS3)
    OCT2DOG3 = calculate_DOG(OCT2GAUS3,OCT2GAUS4)
    OCT2DOG4 = calculate_DOG(OCT2GAUS4,OCT2GAUS5)
    cv.imwrite( "octave2_dog1.jpg", np.asarray(Normalise_Matrix(OCT2DOG1)))
    cv.imwrite( "octave2_dog2.jpg", np.asarray(Normalise_Matrix(OCT2DOG2)))
    cv.imwrite( "octave2_dog3.jpg", np.asarray(Normalise_Matrix(OCT2DOG3)))
    cv.imwrite( "octave2_dog4.jpg", np.asarray(Normalise_Matrix(OCT2DOG4)))
    print("Octave 2 DoG Completed")
    
    #This part of code Locates the maxima and minima to find the keypoints
    OCT2Key1 = calculate_keypoints(OCT2DOG1,OCT2DOG2,OCT2DOG3)
    OCT2Key2 = calculate_keypoints(OCT2DOG2,OCT2DOG3,OCT2DOG4)
    cv.imwrite( "octave2_key1.jpg", np.asarray(Normalise_Matrix(OCT2Key1)))
    cv.imwrite( "octave2_key2.jpg", np.asarray(Normalise_Matrix(OCT2Key2)))
    print("Octave 2 Keypoint images Generated")

    #this part of code points the generated key point value in original image
    OCT2FINAL = mark_keypoints(OCT2Key1,OCT2Key2,second_image)
    cv.imwrite("octave2_final.jpg",np.asarray(OCT2FINAL))
    print("Octave 2 final Image generated")


def octave3_gaussian(third_image):
    # OCTAVE 3 Code starts here
    # Computing the Matrix for the given original image for various sigma values
    # Sigma values = 2.828,4,5.657,8,11.314
    OCT3GAUS1 = AddGaussianFilterToImage(2.828, third_image)
    OCT3GAUS2 = AddGaussianFilterToImage(4, third_image)
    OCT3GAUS3 = AddGaussianFilterToImage(5.657, third_image)
    OCT3GAUS4 = AddGaussianFilterToImage(8, third_image)
    OCT3GAUS5 = AddGaussianFilterToImage(11.314, third_image)

    cv.imwrite( "octave3_Gaus1.jpg", np.asarray(OCT3GAUS1))
    cv.imwrite( "octave3_Gaus2.jpg", np.asarray(OCT3GAUS2))
    cv.imwrite( "octave3_Gaus3.jpg", np.asarray(OCT3GAUS3))
    cv.imwrite( "octave3_Gaus4.jpg", np.asarray(OCT3GAUS4))
    cv.imwrite( "octave3_Gaus5.jpg", np.asarray(OCT3GAUS5))
    print("Octave 3 Gaussian Completed")

    #These function calls calculate the difference of Gaussian
    #Store it in the respective Identifiers
    OCT3DOG1 = calculate_DOG(OCT3GAUS1,OCT3GAUS2)
    OCT3DOG2 = calculate_DOG(OCT3GAUS2,OCT3GAUS3)
    OCT3DOG3 = calculate_DOG(OCT3GAUS3,OCT3GAUS4)
    OCT3DOG4 = calculate_DOG(OCT3GAUS4,OCT3GAUS5)
    cv.imwrite( "octave3_dog1.jpg", np.asarray(Normalise_Matrix(OCT3DOG1)))
    cv.imwrite( "octave3_dog2.jpg", np.asarray(Normalise_Matrix(OCT3DOG2)))
    cv.imwrite( "octave3_dog3.jpg", np.asarray(Normalise_Matrix(OCT3DOG3)))
    cv.imwrite( "octave3_dog4.jpg", np.asarray(Normalise_Matrix(OCT3DOG4)))
    print("Octave 3 DoG Completed")
    
    #This part of code Locates the maxima and minima to find the keypoints
    OCT3Key1 = calculate_keypoints(OCT3DOG1,OCT3DOG2,OCT3DOG3)
    OCT3Key2 = calculate_keypoints(OCT3DOG2,OCT3DOG3,OCT3DOG4)
    cv.imwrite( "octave3_key1.jpg", np.asarray(Normalise_Matrix(OCT3Key1)))
    cv.imwrite( "octave3_key2.jpg", np.asarray(Normalise_Matrix(OCT3Key2)))
    print("Octave 3 Keypoint images Generated")
    
    #this part of code points the generated key point value in original image
    OCT3FINAL = mark_keypoints(OCT3Key1,OCT3Key2,third_image)
    cv.imwrite("octave3_final.jpg",np.asarray(OCT3FINAL))
    print("Octave 3 final Image generated")

def octave4_gaussian(fourth_image): 
    # OCTAVE 4 Code starts here
    # Computing the Matrix for the given original image for various sigma values
    # Sigma values = 5.657,8,11.314,16,22.627
    OCT4GAUS1 = AddGaussianFilterToImage(5.657, fourth_image)
    OCT4GAUS2 = AddGaussianFilterToImage(8, fourth_image)
    OCT4GAUS3 = AddGaussianFilterToImage(11.314, fourth_image)
    OCT4GAUS4 = AddGaussianFilterToImage(16, fourth_image)
    OCT4GAUS5 = AddGaussianFilterToImage(22.627, fourth_image)
  
    cv.imwrite( "octave4_Gaus1.jpg", np.asarray(OCT4GAUS1))
    cv.imwrite( "octave4_Gaus2.jpg", np.asarray(OCT4GAUS2))
    cv.imwrite( "octave4_Gaus3.jpg", np.asarray(OCT4GAUS3))
    cv.imwrite( "octave4_Gaus4.jpg", np.asarray(OCT4GAUS4))
    cv.imwrite( "octave4_Gaus5.jpg", np.asarray(OCT4GAUS5))
    print("Octave 4 Gaussian Completed")
    
    #These function calls calculate the difference of Gaussian
    #Store it in the respective Identifiers
    OCT4DOG1 = calculate_DOG(OCT4GAUS1,OCT4GAUS2)
    OCT4DOG2 = calculate_DOG(OCT4GAUS2,OCT4GAUS3)
    OCT4DOG3 = calculate_DOG(OCT4GAUS3,OCT4GAUS4)
    OCT4DOG4 = calculate_DOG(OCT4GAUS4,OCT4GAUS5)
    cv.imwrite( "octave4_dog1.jpg", np.asarray(Normalise_Matrix(OCT4DOG1)))
    cv.imwrite( "octave4_dog2.jpg", np.asarray(Normalise_Matrix(OCT4DOG2)))
    cv.imwrite( "octave4_dog3.jpg", np.asarray(Normalise_Matrix(OCT4DOG3)))
    cv.imwrite( "octave4_dog4.jpg", np.asarray(Normalise_Matrix(OCT4DOG4)))
    print("Octave 4 DoG Completed")
    
    #This part of code Locates the maxima and minima to find the keypoints
    OCT4Key1 = calculate_keypoints(OCT4DOG1,OCT4DOG2,OCT4DOG3)
    OCT4Key2 = calculate_keypoints(OCT4DOG2,OCT4DOG3,OCT4DOG4)
    cv.imwrite( "octave4_key1.jpg", np.asarray(Normalise_Matrix(OCT4Key1)))
    cv.imwrite( "octave4_key2.jpg", np.asarray(Normalise_Matrix(OCT4Key2)))
    print("Octave 4 Keypoint images Generated")

    #this part of code points the generated key point value in original image
    OCT4FINAL = mark_keypoints(OCT4Key1,OCT4Key2,fourth_image)
    cv.imwrite("octave4_final.jpg",np.asarray(OCT4FINAL))
    print("Octave 4 final Image generated")

# This is the implementation of SIFT algorithm for key point detection
# This contains First 3 Steps of SIFT Algorithm  
# Read the input image in Gray scale format
Color_image = cv.imread('task2.jpg')
INPUT_IMAGE = cv.imread('task2.jpg', cv.IMREAD_GRAYSCALE)
second_image = ScaleDownImage(INPUT_IMAGE)
print(np.asarray(second_image).shape)
third_image = ScaleDownImage(second_image)
print(np.asarray(third_image).shape)
fourth_image = ScaleDownImage(third_image)

print("Process initiated...")
octave1_gaussian(INPUT_IMAGE)
octave2_gaussian(second_image)
octave3_gaussian(third_image)
octave4_gaussian(fourth_image)
print("process completed")


