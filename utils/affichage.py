"""
Module contenant des fonctions utiles pour l'affichage des images notamment.
"""

from math import sqrt, floor
import matplotlib.pyplot as plt

def afficher_liste(liste,x) :
    """
    Entrée : - liste est de la forme : [34, 728, 612, 555] où chaque nombre est le numéro de l'image
             - x est la liste contenant les images
    Sortie : rien
    Affiche les images présentes dans liste
    """
    taille = len(liste)
    sqrtaille = sqrt(taille)
    fig = plt.figure(figsize=(3*floor(sqrtaille)+1, 3*floor(sqrtaille)+1))
    for i,num in enumerate(liste) :
        plt.subplot(floor(sqrtaille)+1, floor(sqrtaille)+1, i+1)
        plt.imshow(x[num-1]);
        plt.title(str(num))
        plt.axis('off')
    plt.show()

def afficher_txt(file_txt,x):
    """
    Entrée : - file_txt est un fichier texte dont chaque ligne est un nombre entre 1 et 50000
             - x est la liste contenant les images
    Sortie : rien
    Affiche les images dont les numéros sont présents dans le fichier texte
    """
    file = open(file_txt,'r')
    lignes = file.readlines()
    liste = []
    for ligne in lignes :
        liste.append(int(ligne.rstrip("\n")))
    afficher_liste(liste,x)