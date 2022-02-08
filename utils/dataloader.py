import cv2
import numpy as np
import os
from matplotlib.path import Path
import json

def mask_to_tensor(mask_path,class_num,bg_num):
    """
    Prend en entrée un masque (n,m) et renvoie un tenseur (n,m,11) accepté par le réseau
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    tensor = np.zeros((img.shape[0],img.shape[1],11))
    tensor[:,:,class_num] = np.where(img>0,1,0)
    tensor[:,:,bg_num] = 1-tensor[:,:,class_num]
    return tensor

def load_data6():
    """
    Renvoie deux tenseurs x et y, qui correspondent aux images que l'on a annotées (x) et aux annotations (y), seulement 6 classes sur les 10
    """
    # On garde les 10 classes, même si on ne prédira que 6 classes
    animals = ["chat","lynx","loup","coyote","jaguar","guepard","chimpanze","orang_outan","hamster","cochon_d_inde"]

    x = np.empty((0,64,64,3))
    y = np.empty((0,64,64,11))

    for (i,animal) in enumerate(animals) :
        folder_name_mask = "masques/"+animal+"_mask"
        folder_name_img = "animals/fromclass/"+animal
        for fichier in os.listdir(folder_name_mask) :
            # Le groupe qui nous a gentiment donné des données annotées a renommé les images. Avec cette boucle conditionnelle, on ne prend pas les données annotées par eux, on prend seulement les nôtres
            if fichier.split("_")[0] not in ["chat","lynx","loup","coyote","jaguar","guepard","chimpanze","orang-outan","hamster","cochon-dinde"]:
                # Process mask
                path_mask = folder_name_mask+"/"+fichier
                mask = mask_to_tensor(path_mask,class_num=i,bg_num=10)
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

def load_data10():
    """
    Renvoie deux tenseurs x et y, qui correspondent aux images que l'on a annotées + celles qui nous ont été données par un autre groupe (x) et aux annotations (y), les 10 classes sont représentées.
    """
    animals = ["chat","lynx","loup","coyote","jaguar","guepard","chimpanze","orang_outan","hamster","cochon_d_inde"]

    x = np.empty((0,64,64,3))
    y = np.empty((0,64,64,11))

    for (i,animal) in enumerate(animals) :
        folder_name_mask = "masques/"+animal+"_mask"
        folder_name_img = "animals/fromclass/"+animal
        for fichier in os.listdir(folder_name_mask) :
            # Process mask
            path_mask = folder_name_mask+"/"+fichier
            mask = mask_to_tensor(path_mask,class_num=i,bg_num=10)
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

def create_mask(poly):

    width, height=64, 64

    #polygon=[(0.1*width, 0.1*height), (0.15*width, 0.7*height), (0.8*width, 0.75*height), (0.72*width, 0.15*height)]
    poly_path=Path(poly)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)

    mask = poly_path.contains_points(coors)
    
    mask = np.transpose(mask.reshape(height, width))
    #plt.imshow(mask.reshape(height, width))
    #plt.show()
    
    return mask

def load_test_data():
    """
    Load les données de test annotées par Axel Carlier dans les tenseurs x et y.
    Auteur : Axel Carlier
    """
    x = np.zeros((100, 64, 64, 3))
    masks = np.zeros((100, 64, 64))

    index_img = 0
    for file in range(10):
        start_image = file*10+1
        end_image = (file+1)*10
        
        json_file = 'annotations' + str(start_image) + '-' + str(end_image) + '.json'
        with open('testprojet/annotations/segmentation/' + json_file, 'r') as f:      
            data = json.load(f)
            
            for i in range(len(data['images'])):
                
                new_mask = 10*np.ones((64,64))
                
                data_img = data['images'][i]
                img_name = data_img['image']
                img = cv2.imread('testprojet/test/' + img_name)
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #plt.imshow(RGB_img)
                #plt.show()
                polygon_annotations = data_img['annotations'][0]['annotations']
                
                for m in range(len(polygon_annotations)):
                    obj_class = polygon_annotations[m]['classId']
                    mask = create_mask(np.array(polygon_annotations[m]['annotation']))
                    new_mask[mask] = obj_class-1
                    
                #plt.imshow(new_mask)
                #plt.show()
                
                
                x[index_img] = RGB_img
                masks[index_img] = new_mask
                index_img += 1
    
    y = np.zeros((100, 64, 64, 11))

    for i in range(masks.shape[0]):
        for c in range(11):
            y[i, :, :, c] = np.where(masks[i]==c, 1, 0)
    
    return x,y