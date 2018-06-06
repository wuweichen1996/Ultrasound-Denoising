import numpy as np
import cv2
import math
import os

def getdatset(N_train,N_test):
    train_xh = []
    train_xl=[]
    test_xh = []
    test_xl=[]
    train_fileDir = "/home/wcwu/Resize_Train_Images/"
    train_noisy_fileDir = "/home/wcwu/Noisy_Train_Images/"
    test_fileDir = "/home/wcwu/Resize_Test_Images/"
    test_noisy_fileDir = "/home/wcwu/Noisy_Test_Images/"

    direction = os.listdir(train_fileDir)
    cnt = 0
    for filename in direction:
        img = cv2.imread(train_fileDir+filename,cv2.CV_8UC1)
        train_xh.append(img)
        noisy_filename = filename.split('.')[0] + '_noisy.png'
        img = cv2.imread(train_noisy_fileDir+noisy_filename,cv2.CV_8UC1)
        train_xl.append(img)
        cnt = cnt + 1
        if cnt == N_train:
            break

    direction = os.listdir(test_fileDir)
    cnt = 0
    for filename in direction:
        img = cv2.imread(test_fileDir + filename, cv2.CV_8UC1)
        test_xh.append(img)
        noisy_filename = filename.split('.')[0] + '_noisy.png'
        img = cv2.imread(test_noisy_fileDir + noisy_filename, cv2.CV_8UC1)
        test_xl.append(img)
        cnt = cnt + 1
        if cnt == N_test:
            break

    train_xh=np.array(train_xh)
    train_xl=np.array(train_xl)
    test_xh=np.array(test_xh)
    test_xl=np.array(test_xl)
    train_xh = np.reshape(train_xh, (train_xh.shape[0], 128, 128,1))
    test_xh = np.reshape(test_xh, (test_xh.shape[0],  128, 128,1))
    train_xl = np.reshape(train_xl, (train_xl.shape[0],  128, 128,1))
    test_xl = np.reshape(test_xl, (test_xl.shape[0],  128, 128,1))
    return  train_xh,test_xh,train_xl,test_xl

def cclt_psnr(ori_img, noisy_img, denoise_img):
    float_type = np.result_type(ori_img.dtype, noisy_img.dtype, np.float32)
    ori_img = ori_img.astype(float_type)
    noisy_img = noisy_img.astype(float_type)
    mse_noisy = np.mean(np.square(ori_img - noisy_img), dtype=np.float64)
    psnr_noisy = 10 * np.log10((255 ** 2) / mse_noisy)

    float_type1 = np.result_type(ori_img.dtype, denoise_img.dtype, np.float32)
    denoise_img1 = denoise_img.astype(float_type1)
    mse_noisy1 = np.mean(np.square(ori_img - denoise_img1), dtype=np.float64)
    psnr_denoise = 10 * np.log10((255 ** 2) / mse_noisy1)
    return psnr_noisy, psnr_denoise
