# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 15:37:42 2023

@author: paur
"""

import os
import random
import shutil


folder_name = "data_con"



def create_folders(path,train_dir, labels, num):
    """
    Creates folders for training data based on labels and number of samples.
    Args:
        train_dir: Directory containing training images
        labels: List of labels 
        num: Number to append to folder name
    Returns: 
        None: No value is returned
    - Counts number of samples in training directory
    - Creates folder with path + folder_name + str(num)  
    - Creates train subdirectory to contain images
    - Loops through labels to create individual label folders"""
    number_of_samples=len(os.listdir(train_dir))
    print(number_of_samples)
    path_dest=path+folder_name+str(num)
    os.makedirs(path_dest)
    train_dest=path_dest + "/" + "train"
    os.makedirs(train_dest)
    for i in labels:
        label_train_path=train_dest + "/" + str(i)
        os.makedirs(label_train_path)

def small_batch(path,train_dir, labels, num):
    train_dest = path+folder_name + str(num) + "/train"
    for i in labels:
        train_path=train_dest + "/" + i
        file_names_train = os.listdir(train_dir+"/" + i)
        number_train = len(file_names_train)
        target_images_batch = random.sample(file_names_train,int((num*number_train)/100))
        for j, img in enumerate(target_images_batch):
            shutil.copy(train_dir + "/" + str(i) + "/" + str(img),train_path + "/" + img)
            


def get_images(path,train_dir, labels,batch,num):
    train_dest = path+folder_name + str(num) + "/train"
    batch_source = path+folder_name + str(batch) + "/train"
    for i in labels:
        batch_path=batch_source + "/" + i
        file_names_batch = os.listdir(batch_path)
        for j, img in enumerate(file_names_batch):
            shutil.copy(batch_path + "/" + img, train_dest + "/" + str(i)+ "/" + img)
    for i in labels:
        train_path=train_dest + "/" + i
        file_names_train = os.listdir(train_dir +"/" + i)
        number_train = len(file_names_train)
        fill_images(train_dir,train_path,file_names_batch, file_names_train,num,number_train,i,batch)


def fill_images (train_dir,images, batch_images,train_images, num,number_train,i,batch):
    max_values = len(os.listdir(images))
    flag = 0
    target_images = random.sample(train_images,int(((num-batch)*number_train)/100))
    total_images=int((num*number_train)/100)
    for j, img in enumerate(target_images):
        for l, img_1 in enumerate(batch_images):
            if img != img_1:
                flag=1
        if flag==1:
            if max_values >= total_images:
                break
            shutil.copy(train_dir +"/" + str(i)+"/" + str(img),images + "/" + img)
            max_values +=1
        flag = 0
    if max_values < total_images:
        fill_images(train_dir,images, os.listdir(images),train_images, num,number_train,i,batch)

