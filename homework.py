import cv2 as cv
import numpy as np

img = cv.imread("ushape.png",0)
shape = img.shape
print(shape)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5),anchor=(3,3))
kernel[0,1]=1
kernel[0,3]=1
kernel[4,1]=1
kernel[4,3]=1
opening = cv.dilate(img,kernel,iterations=3)
kernel2 = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))

opening1 = cv.dilate(opening,kernel2,iterations=3)
cv.imwrite("dilation.jpg",opening1)

kernel3 = cv.getStructuringElement(cv.MORPH_RECT,(5,5),anchor= (4,4))
opening2 = cv.erode(img,kernel3,iterations=4)
cv.imwrite("dilationa.jpg",opening2)

kernel4 = cv.getStructuringElement(cv.MORPH_RECT,(15,5),anchor=(13,2))
opening3 = cv.erode(img,kernel4,iterations=4)
cv.imwrite("dilationb.jpg",opening3)
