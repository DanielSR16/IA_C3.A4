from tensorflow import keras
import cv2 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
   
from tkinter import *
   
from tkinter import filedialog 
modelo = keras.models.load_model("modelo/modelo.h5")

def modelo_ejecutar(ruta):
    image = cv2.resize(cv2.imread(rf"{ruta}"),(200,200))

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    probar_img = tf.cast(image,tf.float32)
    probar_img /= 255

    array_img = np.array([probar_img])


    resul = modelo.predict(array_img)

    res = np.argmax(resul)


    nombres = ['Cuchara','Cuchillo','Tenedor']
    # print(resul)
    # print(nombres[res])
    
    plt.figure()
    plt.imshow(probar_img, cmap=plt.cm.binary)
    plt.colorbar()
    plt.title('Resultado: '+str(nombres[res]))
    plt.grid(False)
    plt.show()

def browseFiles(): 
    filename = filedialog.askopenfilename(initialdir = "/", 
    title = "Seleccionar Archivo", 
    filetypes = ( ("Todos los archivos", 
            "*.*"),("Texto", 
            "*.txt*"),
           )) 
       
    
    modelo_ejecutar(filename)
       
       
                                                                                                   
window = Tk() 
   
window.title('Analizador de imagenes') 
   
window.geometry("500x200") 
   
window.config(background = "white") 
   
label_file_explorer = Label(window,  
text = "Analizador de imagenes de cubiertos cucharas , tenedores y cuchillos", 
width = 100, height = 4,  
fg = "blue",
) 
   
       
button_explore = Button(window,  
                        text = "Explorar", 
                        command = browseFiles)  
   
 
   
label_file_explorer.place(relx = 0.5,
                   rely = 0.1,
                   anchor = 'center') 
   
button_explore.place(relx = 0.5,
                   rely = 0.5,
                   anchor = 'center') 
   

   
window.mainloop() 