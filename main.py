# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:55:32 2023

@author: Eason
"""
from  matplotlib import pyplot as plt
import cv2
import numpy as np
import math
from skimage import transform



#invert
def invert(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = -image[i][j][k]+255
    
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('invert')
    plt.show()
    print("original shape: ",image.shape)
    print("invert shape: ",res.shape)
    return res

#darken
def darken(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = image[i][j][k]-128
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('darken')
    plt.show()
    print("original shape: ",image.shape)
    print("darken shape: ",res.shape)
    return res

#lighten
def lighten(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = image[i][j][k]+128
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('lighten')
    plt.show()
    print("original shape: ",image.shape)
    print("lighten shape: ",res.shape)
    return res

#lower contrast
def lower_contrast(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = image[i][j][k]//2
    res = res.astype(np.uint8)
    res = np.clip(res,0,255)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('lower contrast')
    plt.show()
    print("original shape: ",image.shape)
    print("lower contrast shape: ",res.shape)
    return res

#raise contrast
def raise_contrast(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = image[i][j][k]*2
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('raise contrast')
    plt.show()
    print("original shape: ",image.shape)
    print("raise contrast shape: ",res.shape)
    return res

#non-linear raise contrast
def non_linear_raise_contrast(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = (image[i][j][k]/255)**2*255
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('non-linear raise contrast')
    plt.show()
    print("original shape: ",image.shape)
    print("non-linear raise contrast shape: ",res.shape)
    return res

#non-linear lower contrast
def non_linear_lower_contrast(image):
    res = np.zeros_like(image).astype(np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                res[i][j][k] = (image[i][j][k]/255)**(1/3)*255
    res = np.clip(res,0,255)
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('non-linear lower contrast')
    plt.show()
    print("original shape: ",image.shape)
    print("non-linear lower contrast shape: ",res.shape)
    return res


def get_kernel(kernel_name, size = 7): # here we set the kernel_size to 7, when you call this function without kernel_size, the default is 7
  if kernel_name == 'gaussian':
    sigma = 0.3*((size-1)*0.5 - 1) + 0.8
    Gaussian=[]
    x, y = np.mgrid[-size:size+1, -size:size+1]
    Gaussian = np.exp(-((x**2 + y**2)/(2*sigma**2)))
    return Gaussian / Gaussian.sum()

  if kernel_name == 'sharpening':
    return np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

  if kernel_name == 'mean':
    Mean_Kernel = np.zeros((size, size))
    val = 1 / (size ** 2)
    for i in range(size):
        for j in range(size):
            Mean_Kernel[i][j] = val
    return Mean_Kernel

  if kernel_name == 'shift':
    Shift_Kernel = np.zeros((size,size))
    Shift_Kernel[size//2][0] = 1
    return Shift_Kernel

  if kernel_name == 'dilation':
    return np.ones((size,size))

  if kernel_name == 'erosion':
    return np.array([[1,-1,1],[0,-1,0],[1,-1,1]])
    
  if kernel_name == 'sobel_dx':
      return np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
  
  if kernel_name == 'sobel_dy':
      return np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
  
  if kernel_name == 'dilation':
      return np.ones((size,size))
  
  if kernel_name == 'erosion':
      return np.array([[1,-1,1],[0,-1,0],[1,-1,1]])

  
def conv(img,kernel,style='constant'):
    k_height, k_width = kernel.shape
    pad_height = (k_height ) // 2
    pad_width = (k_width ) // 2
    if img.ndim == 3:
        padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
        output = np.zeros_like(img).astype(np.int32)
        for k in range(img.shape[2]):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    output[i,j,k] = np.sum(padded_image[i:i+k_height, j:j+k_width, k] * kernel)
        output = np.clip(output,0,255)
        output = output.astype(np.uint8)
    elif img.ndim == 2:
        padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        output = np.zeros_like(img).astype(np.int32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                output[i,j] = np.sum(padded_image[i:i+k_height, j:j+k_width] * kernel)
        output = np.clip(output,0,255)
        output = output.astype(np.uint8)
    return output
    


#gaussian
def gaussian(image, size = 7):
    kernel = get_kernel('gaussian', size = 7)
    res = conv(image,kernel,style='constant')
    
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('gaussian')
    plt.show()
    print("original shape: ",image.shape)
    print("gaussian shape: ",res.shape)
    return res

#sharpening
def sharpening(image, size = 7):
    kernel = get_kernel('sharpening', size = 7)
    res = conv(image,kernel,style='constant')
    
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('sharpening')
    plt.show()
    print("original shape: ",image.shape)
    print("sharpening shape: ",res.shape)
    return res

#mean
def mean(image, size = 7):
    kernel = get_kernel('mean', size = 7)
    res = conv(image,kernel,style='constant')
    
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('mean')
    plt.show()
    print("original shape: ",image.shape)
    print("mean shape: ",res.shape)
    return res

#shift
def shift(image, size = 7):
    kernel = get_kernel('shift', size = 10)
    res = conv(image,kernel,style='constant')
    
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('shift')
    plt.show()
    print("original shape: ",image.shape)
    print("shift shape: ",res.shape)

#sobel
def sobel(image, size = 7):
    kernel_dx = get_kernel('sobel_dx', size = 7)
    kernel_dy = get_kernel('sobel_dy', size = 7)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res_dx = conv(img,kernel_dx,style='constant')
    res_dy = conv(img,kernel_dy,style='constant')
    
    res_dx = res_dx.astype(np.uint8)
    res_dy = res_dx.astype(np.uint8)
    res = (res_dx**2+res_dy**2)**.5
    res = res.astype(np.uint8)
    
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 4, 2)
    plt.imshow(res_dx,cmap='gray')
    plt.title('sobel_dx')
    
    
    plt.subplot(1, 4, 3)
    plt.imshow(res_dy,cmap='gray')
    plt.title('sobel_dy')
    
    plt.subplot(1, 4, 4)
    plt.imshow(res_dy,cmap='gray')
    plt.title('sobel')
    plt.show()
    
    print("original shape: ",image.shape)
    print("sobel_dx shape: ",res_dx.shape)
    print("sobel_dy shape: ",res_dy.shape)
    print("sobel shape: ",res.shape)
    return res_dx,res_dy,res


#thresholding
def thresholding(image,threshold_value = 125):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = np.zeros_like(img).astype(np.int32)
    res[img > threshold_value] = 255
    
    res = res.astype(np.uint8)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res,cmap='gray')
    plt.title('thresholding')
    plt.show()
    print("original shape: ",image.shape)
    print("thresholding shape: ",res.shape)

#downsampling
def downsampling(image,factor):
    length , width = image.shape[0],image.shape[1]
    res =  image[:length-factor:factor, :width-factor:factor,:]
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('downsampling')
    plt.show()
    print("original shape: ",image.shape)
    print("downsampling shape: ",res.shape)
    return res

#unsampling
def unsampling(image,factor):
    [length , width , height] = image.shape
    
    res = np.zeros((length*factor, width*factor, height), dtype=image.dtype)
    for i in range(length):
        for j in range(width):
            res[i*factor:i*factor+factor, j*factor:j*factor+factor,:] = image[i, j, :]

    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res)
    plt.title('unsampling')
    plt.show()
    print("original shape: ",image.shape)
    print("unsampling shape: ",res.shape)
    return res

#dilation
def dilation(image):
    kernel = get_kernel('dilation', size = 7)
    res = conv(image,kernel,style='constant')
    
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap = 'gray')
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res,cmap = 'gray')
    plt.title('dilation')
    plt.show()
    print("original shape: ",image.shape)
    print("dilation shape: ",res.shape)
    return res

#erosion
def erosion(image): 
    
    kernel = get_kernel('erosion', size = 7)
    res = conv(image,kernel,style='constant')
    
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap = 'gray')
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res,cmap = 'gray')
    plt.title('erosion')
    plt.show()
    print("original shape: ",image.shape)
    print("erosion shape: ",res.shape)
    return res

#reflection
def reflection(image):   
    length , width = image.shape[0],image.shape[1]
    res = np.zeros_like(image)
    for i in range(length):
        for j in range(width): #new loc[x,y] = [-x,y]
            res[i, -j+width-1] = image[i, j]
    
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap = 'gray')
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res,cmap = 'gray')
    plt.title('reflection')
    plt.show()
    print("original shape: ",image.shape)
    print("reflection shape: ",res.shape)
    return res

#rotate
def rotate(image,degree,cut = 'yes'):   
    [length , width ] = image.shape
    grid = np.tile(image, (3, 3))
    res = np.ones(grid.shape)*255
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if i+length//2<grid.shape[0] and j+width//2<grid.shape[1]:
                res[i+length//2,j+width//2]=grid[i,j]
    res = transform.rotate(res, degree, resize=False)
    res = res[int(length*1.5):int(length*2.5),int(width*1.5):int(width*2.5)]
    
    if cut == 'yes':
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if i<=j*math.tan(math.radians(abs(degree))):
                    res[i,j] = 255
    
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap = 'gray')
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res,cmap = 'gray')
    plt.title('rotate')
    plt.show()
    print("original shape: ",image.shape)
    print("rotate shape: ",res.shape)
    return res

#scale
def scale(image,magnitude): 
    [length , width ] = image.shape
    res = np.zeros((length,width*magnitude))
    for i in range(length):
        for j in range(width): #new loc[x,y] = [-x,y]
            res[i, j*magnitude:j*magnitude+magnitude] = image[i, j]
    res = res[:,:width]

    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image,cmap = 'gray')
    plt.title('original')
    
    plt.subplot(1, 2, 2)
    plt.imshow(res,cmap = 'gray')
    plt.title('scale')
    plt.show()
    print("original shape: ",image.shape)
    print("scale shape: ",res.shape)
    return res

if __name__=='__main__':
    image = cv2.imread("C:/Users/88696/Desktop/homework/kookaburra.jpg")
    image = image[:,:,::-1]
    
    #point processing
    invert(image)
    darken(image)
    lighten(image)
    lower_contrast(image)
    raise_contrast(image)
    non_linear_raise_contrast(image)
    non_linear_lower_contrast(image)
    
    #kernel processing
    gaussian(image, size = 7)
    sharpening(image)
    mean(image, size = 7)
    shift(image, size = 12)
    sobel(image)
    thresholding(image,125)
    downsample_img = downsampling(image,8)
    unsampling(downsample_img,8)
    
    #morphology
    image = cv2.imread("C:/Users/88696/Desktop/homework/nicework.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dilation(image)
    erosion(image)
    reflection(image)
    rotate(image,-25,cut = 'no')
    scale(image,2)
