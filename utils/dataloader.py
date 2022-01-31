import cv2
import numpy as np
import os

def mask_to_tensor(mask_path,class_num,bg_num):
    """
    Prend en entrée un masque (n,m) et renvoie un tenseur (n,m,11) accepté par le réseau
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    tensor = np.zeros((img.shape[0],img.shape[1],11))
    tensor[:,:,class_num] = np.where(img>0,1,0)
    tensor[:,:,bg_num] = 1-tensor[:,:,class_num]
    return tensor

def load_data():
    animals = ["chimpanze","coyote","guepard","jaguar","loup","orang_outan"]
    classes = [6,3,5,4,2,7]

    x = np.empty((0,64,64,3))
    y = np.empty((0,64,64,11))

    for (i,animal) in enumerate(animals) :
        folder_name_mask = "masques/"+animal+"_mask"
        folder_name_img = "animals/fromclass/"+animal
        for fichier in os.listdir(folder_name_mask) :
            # Process mask
            path_mask = folder_name_mask+"/"+fichier
            mask = mask_to_tensor(path_mask,class_num=classes[i],bg_num=10)
            mask_extend = np.expand_dims(mask,0)
            y = np.append(y,mask_extend,0)
            # Process image
            img_name = fichier.split(".")[0]+".jpg"
            path_img = folder_name_img+"/"+img_name
            img = cv2.imread(path_img)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_extend = np.expand_dims(img,0)
            x = np.append(x,img_extend,0)
    return x,y

def permute_y(y0,y1):
    """
    Permute the columns in the 4th axis of y0 to match the indexation of y1
    """
    ind0 = np.arange(11)
    ind1 = np.array([5,6,3,4,8])