
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# In[2]:

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


# In[3]:


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


# In[4]:


def hough_line(img):
      # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(img.shape[0] * img.shape[0] + img.shape[1] * img.shape[1]))   # max_dist
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * int(diag_len), len(thetas)))
    rhos=[]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]!= 0.:
                for x in range(len(thetas)):
                    # Calculate rho. diag_len is added for a positive index
                    rho = round(i * np.cos(thetas[x]) + j * np.sin(thetas[x]))+diag_len
                    rhos.append(rho)
                    accumulator[int(rho), int(x)] += 1

    return accumulator, thetas, rhos


# In[75]:


def minimise_no_rhos(rho,theta):
    rho_op =[] 
    theta_op = []
    count= 0
    array = []
    for x in rho:
        count+=1
        if len(rho_op) == 0:
            rho_op.append(x)
        else:
            if (x - rho[count-1]) < 10:
                array.append(x)
            else:
                rho_op.append(np.median(array))
                theta_op.append(theta[count-1])
                array = []
                
    return rho_op,theta_op


# In[6]:


img = cv2.imread('original_imgs/hough.jpg')
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([170,110,110])
upper_red = np.array([180,150,150])
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

output_hsv = hsv_img.copy()
output_hsv[np.where(mask1==0)] = 0

cv2.imwrite("onlyred.jpg",output_hsv)


# In[7]:


edges = detectEdges(output_hsv)
cv2.imwrite("red_edge.jpg",edges)


# In[76]:


accumulators,thetas,rhos = hough_line(edges)
print(accumulators.shape)
# code to get the indices which are above the range
idx = np.argmax(accumulators)
print(idx)
print(accumulators[int(idx/accumulators.shape[1]),int(idx%accumulators.shape[1])])
mask = [accumulators > 150.] [0] * 1.
accum = accumulators * mask
rho = []
theta = []
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i,j]==1:
            rho.append(i)
            theta.append(thetas[j])         


# In[77]:


print(rho)
print(theta)


# In[79]:


r,t= minimise_no_rhos(rho,theta)


# In[33]:


rho = [257,258,352,353,445,446,541,542,636,637,734,733]
theta = [-1.53588974175501 ,-1.53588974175501,-1.53588974175501, -1.53588974175501,-1.53588974175501,-1.53588974175501]
#gray_img = cv2.imread('original_imgs/hough.jpg', cv2.IMREAD_GRAYSCALE)
dup = np.copy(edges)
#gs = gray_img.shape
diag_len = np.ceil(np.sqrt(dup.shape[0] *dup.shape[0]+ dup.shape[1] * dup.shape[1]))
indices = []
for i in range(dup.shape[0]):
    for j in range(dup.shape[1]):
        for t in theta:
            rho_dup = round(i * np.cos(t) + j * np.sin(t))+diag_len
            #print(rho_dup, rho)
            for r in rho:
                if r == rho_dup:
                    indices.append([i, j])            


# In[34]:


gray_img = cv2.imread('original_imgs/hough.jpg')
output = np.zeros((gray_img.shape[0],gray_img.shape[1]))

for i in indices:
    gray_img[i[0],i[1]] = (0,255,0)

cv2.imwrite("output/red_line.jpg",gray_img)


# In[49]:


img_b = cv2.imread('original_imgs/hough.jpg')
hsv_img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2HSV)
lower_blue = np.array([95,60,110],np.uint8)
upper_blue = np.array([110,140,150],np.uint8)
mask1 = cv2.inRange(hsv_img_b, lower_blue, upper_blue)

output_hsv_b = hsv_img_b.copy()
output_hsv_b[np.where(mask1==0)] = 0

cv2.imwrite("onlyblue.jpg",output_hsv_b)


# In[50]:


edges_b = detectEdges(output_hsv_b)
cv2.imwrite("blue_edge.jpg",edges_b)


# In[62]:


accumulators_b,thetas_b,rhos_b = hough_line(edges_b)
print(accumulators_b.shape)
# code to get the indices which are above the range
idx_b = np.argmax(accumulators_b)
print(idx_b)
print(accumulators_b[int(idx_b/accumulators_b.shape[1]),int(idx_b%accumulators_b.shape[1])])
mask_b = [accumulators_b > 125.] [0] * 1.
accum_b = accumulators_b * mask_b
rho_b = []
theta_b = []
for i in range(mask_b.shape[0]):
    for j in range(mask_b.shape[1]):
        if mask_b[i,j]==1:
            rho_b.append(i)
            theta_b.append(thetas_b[j])


# In[80]:


r_b,t_b= minimise_no_rhos(rho_b,theta_b)


# In[65]:


rho_b = [489,490,564,565,637,638,708,709,779,780,853,854,930,931]
theta_b =[-0.9424777960769379]
#gray_img = cv2.imread('original_imgs/hough.jpg', cv2.IMREAD_GRAYSCALE)
dup_b = np.copy(edges_b)
#gs = gray_img.shape
diag_len_b = np.ceil(np.sqrt(dup_b.shape[0] *dup_b.shape[0]+ dup_b.shape[1] * dup_b.shape[1]))
indices_b = []
for i in range(dup_b.shape[0]):
    for j in range(dup_b.shape[1]):
        for t in theta_b:
            rho_dup = round(i * np.cos(t) + j * np.sin(t))+diag_len_b
            #print(rho_dup, rho_b)
            for r in rho_b:
                if r == rho_dup:
                    indices_b.append([i, j])  


# In[66]:


gray_img_b = cv2.imread('original_imgs/hough.jpg')
#output = np.zeros((gray_img_b.shape[0],gray_img_b.shape[1]))

for i in indices_b:
    gray_img_b[i[0],i[1]] = (0,255,0)

cv2.imwrite("output/blue_line.jpg",gray_img_b)


# In[ ]:


img = cv2.imread('original_imgs/hough.jpg')
hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower_red = np.array([0,20,50])
upper_red = np.array([60,255,255])
mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

output_hsv = hsv_img.copy()
output_hsv[np.where(mask1==0)] = 0

cv2.imwrite("onlycoin.jpg",output_hsv)


# In[ ]:


edges_b = detectEdges(output_hsv)
cv2.imwrite("coin_Edge.jpg",edges_b)


# In[ ]:


def HoughCircles(input): 
    circles=[]
    rows = input.shape[0] 
    cols = input.shape[1] 
    
    sinang = dict() 
    cosang = dict() 
    
    for angle in range(0,360): 
        sinang[angle] = np.sin(angle * np.pi/180) 
        cosang[angle] = np.cos(angle * np.pi/180) 
            
    radius = [i for i in range(10,70)]

    threshold = 190 
    
    for r in radius:
        acc_cells = np.full((rows,cols),fill_value=0,dtype=np.uint64)
         
        for x in range(rows): 
            for y in range(cols): 
                if input[x][y] != 0:# edge 
                    for angle in range(0,360): 
                        b = y - round(r * sinang[angle]) 
                        a = x - round(r * cosang[angle]) 
                        if a >= 0 and a < rows and b >= 0 and b < cols: 
                            acc_cells[int(a)][int(b)] += 1
                             
        print('For radius: ',r)
        acc_cell_max = np.amax(acc_cells)
        print('max acc value: ',acc_cell_max)
        
        if(acc_cell_max > 500):  

            print("Detecting the circles for radius: ",r)       
            
            # Initial threshold
            acc_cells[acc_cells < 150] = 0  
               
            # find the circles for this radius 
            for i in range(rows): 
                for j in range(cols): 
                    if(i > 0 and j > 0 and i < rows-1 and j < cols-1 and acc_cells[i][j] >= 150):
                        avg_sum = np.float32((acc_cells[i][j]+acc_cells[i-1][j]+acc_cells[i+1][j]+acc_cells[i][j-1]+acc_cells[i][j+1]+acc_cells[i-1][j-1]+acc_cells[i-1][j+1]+acc_cells[i+1][j-1]+acc_cells[i+1][j+1])/9) 
                        print("Intermediate avg_sum: ",avg_sum)
                        if(avg_sum >= 33):
                            print("For radius: ",r,"average: ",avg_sum,"\n")
                            circles.append((i,j,r))
                            acc_cells[i:i+5,j:j+7] = 0
                 
    return circles


# In[ ]:


# Detect Circle 
circles = HoughCircles(edges_b)  

# Print the output
for vertex in circles:
    cv2.circle(img,(vertex[1],vertex[0]),vertex[2],(0,255,0),1)
         
cv2.imwrite('Circle_Detected_Image.jpg',img) 

