"""
Module contenant des fonctions utiles pour déplacer des fichiers dans d'autres répertoires notamment.
"""

from shutil import copy
import os

def txt_to_folder() :
    """
    Copie toutes les images des fichiers txt [animal].txt se trouvant de imagestolabel vers les dossiers fromclass/[animal]/
    """
    for fichier in os.listdir('imagestolabel/') :
        name = fichier.split('.')[0]
        directory = 'animals/fromclass/'+ name
        file = open('imagestolabel/'+fichier,'r')
        lignes = file.readlines()
        for ligne in lignes :
            numero = ligne.rstrip("\n")
            copy('animals/unlabelled/img'+numero+'.jpg',directory)