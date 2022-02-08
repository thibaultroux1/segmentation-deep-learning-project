import numpy as np
import seaborn
import matplotlib.pyplot as plt

def confusion_matrix(model,x,y):
    y_pred = model.predict(x)

    matrix = np.zeros((11,11))

    count_pix_class = np.zeros(11)

    for k in range(len(y_pred)):
        ind_ligne = np.argmax(y[k],2)
        ind_colonne = np.argmax(y_pred[k],2)
        ind_ligne_v = ind_ligne.flatten()
        ind_colonne_v =ind_colonne.flatten()
        for i in range(len(ind_ligne_v)):
            matrix[ind_ligne_v[i],ind_colonne_v[i]]+=1
            count_pix_class[ind_ligne_v[i]]+=1

    for i in range(matrix.shape[0]):
        matrix[i,:] = matrix[i,:]/count_pix_class[i]
    
    labels = ["chat","lynx","loup","coyote","jaguar","guepard","chimpanze","orang_outan","hamster","cochon_d_inde","background"]
    plt.figure()
    s = seaborn.heatmap(matrix,annot=True)
    s.set_xlabel("Prédictions",fontsize=20)
    s.set_ylabel("Réel",fontsize=20)
    s.set_xticklabels(labels,fontsize=15)
    s.set_yticklabels(labels,fontsize=15,rotation=0)
    s.set_title("Matrice de confusion",fontsize=30)

def plot_training_analysis(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))
  
  plt.plot(epochs, acc, 'b', linestyle="--",label='Training accuracy')
  plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'b', linestyle="--",label='Training loss')
  plt.plot(epochs, val_loss,'g', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()