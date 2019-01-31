# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matplotlib.image import imread
from PIL import Image

import sys


def create_model():   
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    return model


def load_image(path):  
    #carica un immagine 
    img=imread(path)
    #rende l'immagine in bianco e nero
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    
    #Normalizza l'immagine se non e' già normalizzata
    if(np.max(img)>1):
        img=img/255

    #inverte i colori (nel data set lo sfondo e' nero e la scritta in bianco)
    img=1-img

    return resize_with_cv2(img,(28,28))

def resize_with_cv2(img,size):
    from cv2 import resize
    return resize(img, dsize=size)

def show_img(img):
    plt.figure()
    plt.imshow(img,cmap='Greys')
    plt.show()

def mnist_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    test_images=test_images/255.0
    return (test_images,test_labels)




if __name__ == '__main__':
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    img=load_image(sys.argv[1])

    
    model = create_model()
    (imgs,lbl)=mnist_data()
    
                
    saver=tf.train.Saver()

    with tf.Session() as sess:
        
        saver.restore(sess,'mnistFashion50/mnistFashion50.ckpt')
        # access a variable from the saved Graph, and so on:

        print("\n\nRete mnistFashion50 caricata con successo\n\n")
        
        #test_loss, test_acc = model.evaluate(imgs, lbl)
        #print('Test accuracy:', test_acc)
        #trasformo l'immagine in un array poiche' la rete consuma array
        inputs= np.array([img])
      
        prediction=model.predict(inputs)

        print("Il numero mostrato e' un ")

        
        print(class_names[np.argmax(prediction)])
        print("Con probabilita': ")
        print(prediction[0,np.argmax(prediction)])

        show_img(img)

        #Provo a predirre delle immagini prese da mnist

        

        
        prediction2=model.predict(imgs)
        fig=plt.figure(figsize=(25,25))
        row=10
        col=5
        for i in range(1,row*col+1):
            j=i-1
            index=np.argmax(prediction2[j])
            fig.add_subplot(row,col,i)
            plt.imshow(imgs[j],cmap='Greys')
            plt.ylabel("Predizione "+str(j)+": "+class_names[index],fontsize=5)
        plt.show()
            

