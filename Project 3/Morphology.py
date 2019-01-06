import cv2
import numpy as np

# funtion to apply eclipse kernel for an image
def multiply_kernel(matrix, row, col):
    matrix[row,col] = 255
    if row-1 >=0:
        matrix[row-1, col]=255

    if row+1 < matrix.shape[0] :
        matrix[row+1,col] = 255

    if col-1 >= 0:
        matrix[row,col-1] = 255

    if col+1 < matrix.shape[1]:
        matrix[row,col+1] = 255

    return matrix

def dilation(image,kernel):
    row = image.shape[0]
    col = image.shape[1]
    output = np.copy(image)
    for i in range(row):
        for j in range(col):
            if image[i, j] == 255:
                output = multiply_kernel(output,i,j)
    return output

img = cv2.imread('original_imgs/noise.jpg',0)

#kernel = np.ones((3,3),np.uint8)
kernel = [[0,1,0],[1,1,1],[0,1,0]]
kernel =np.array(kernel)

#dilation = dilation(img,kernel)

def invertImage(img):
    row = img.shape[0]
    col = img.shape[1]
    output = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            if img[i,j] == 255:
                output[i,j] = 0
            elif img[i,j] == 0:
                output[i,j] = 255
    
    return output

def erosion(img,kernel):
    invertedimg =  invertImage(img)
    dilated = dilation(invertedimg,kernel)
    output = invertImage(dilated)
    return output


eroded_img = erosion(img,kernel)
cv2.imwrite("erosion_self.jpg",eroded_img)

def opening(img,kernel):
    return dilation(erosion(img,kernel),kernel)

def closing(img,kernel):
    return erosion(dilation(img,kernel),kernel)

def boundary(img,kernel):
    return img - erosion(img,kernel)

res_noise1 = opening(img,kernel)
res_noise1 = closing(res_noise1,kernel)
cv2.imwrite("output/res_noise1.jpg",res_noise1)

res_noise2 = closing(img,kernel)
res_noise2 = opening(res_noise2,kernel)
cv2.imwrite("output/res_noise2.jpg",res_noise2)

bound1 = boundary(res_noise1,kernel)
cv2.imwrite("output/res_bound1.jpg",bound1)

bound2 = boundary(res_noise2,kernel)
cv2.imwrite("output/res_bound2.jpg",bound2)

count = 0
for i in range(res_noise1.shape[0]):
    for j in range(res_noise1.shape[1]):
        if res_noise1[i][j] != res_noise2[i][j]:
            count +=1

print(count)
        
