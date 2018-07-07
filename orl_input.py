import cv2 as cv
import numpy as np
import os
import random
import Max_Min

#数据库path
data_path = r'C:\Users\Administrator\Desktop\ORL'

def input_orl2d(train_size):
    path = data_path
    test_size=10-train_size
    train_num=40*train_size
    test_num=40*test_size

    pathlist1=os.listdir(path) 

    namelist=[None]*40
    for i in range(len(pathlist1)):
        pathlist2=os.listdir(path+'\\'+pathlist1[i])
        bmppath=[]
        for img in pathlist2:
            bmppath.append(path+'\\'+pathlist1[i]+'\\'+img)
        namelist[i]=bmppath

    #随机产生训练集和测试集
    '''simple_num=range(0,10)
    random_index=random.sample(simple_num,10)'''
    random_index = [1, 4, 9, 0, 5, 2, 8, 6, 3, 7]
    train_index=random_index[0:train_size]
    test_index=random_index[train_size:10]
    train_dir=[None]*40
    test_dir=[None]*40
    for i in range(len(namelist)):
        pre_train=[]#随机训练集
        for d1 in train_index:
            pre_train.append(namelist[i][d1])
        train_dir[i]=pre_train
        pre_test=[]#随机测试集
        for d2 in test_index:
            pre_test.append(namelist[i][d2])
        test_dir[i]=pre_test    
        
        

    #图像矩阵
    train_mat=np.empty((train_num,112,92,1))
    test_mat=np.empty((test_num,112,92,1))
    train_lab=np.zeros((train_num,40))
    test_lab=np.zeros((test_num,40))
    #训练矩阵
    num=0
    for i in range(len(train_dir)):     
        for j in range(len(train_dir[i])):
           
            img=cv.imread(train_dir[i][j],0)
            n_img=Max_Min.convert_Max_Min(img)
            train_mat[num,:,:,0]=n_img
            train_lab[num,i]=1
            num+=1

            
    #测试矩阵
    num=0
    for i in range(len(test_dir)):     
        for j in range(len(test_dir[i])):
         
            img=cv.imread(test_dir[i][j],0)
            n_img=Max_Min.convert_Max_Min(img)
            test_mat[num,:,:,0]=n_img
            test_lab[num,i]=1
            num+=1


    return train_mat,train_lab,test_mat,test_lab

if __name__ == '__main__':
    input_orl2d()
