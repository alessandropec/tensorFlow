# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



train_images=train_images/250.0
test_images=test_images/250.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

saver = tf.train.Saver()
with tf.Session() as sess:
    model.fit(train_images, train_labels, epochs=50)
    save_path = saver.save(sess, "./mnistFashion50/mnistFashion50.ckpt")
    print("Model saved in path: %s" % save_path)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    predictions = model.predict(test_images)
    fig=plt.figure(figsize=(25,25))
    row=5
    col=5
    for i in range(1,row*col+1):
        j=i-1
        index=np.argmax(predictions[j])
        fig.add_subplot(row,col,i)
        plt.imshow(test_images[j],cmap='Greys')
        plt.ylabel("Predizione "+str(j)+": "+class_names[index],fontsize=10)
    plt.show()












    


