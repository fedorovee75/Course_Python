#Classifier image (10 class, image 28x28)
#Create CNN (first variant) 
import numpy, tensorflow
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
#x_train.shape=(60000, 28, 28), x_test.shape=(60000, 28, 28)
#y_train.shape=(60000,), y_test.shape=(10000,)
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
#x_train.shape=(60000, 28, 28, 1), 1 - for grayscale, 3 - for RGB
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255 
#x_test.shape=(10000, 28, 28, 1), 1 - for grayscale, 3 - for RGB
y_train = tensorflow.keras.utils.to_categorical(y_train)
#y_train.shape=(60000, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test)
#y_test.shape=(60000, 10)
inputs=tensorflow.keras.Input(shape=(28, 28, 1))
hidden1 = tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs) # filters = convolution planes
hidden2 = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden1)
hidden3 = tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(hidden2)
hidden4 = tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2))(hidden3)
hidden5 = tensorflow.keras.layers.Flatten()(hidden4) # transform to 1D layer
hidden6 = tensorflow.keras.layers.Dropout(rate=0.5)(hidden5)
hidden7 = tensorflow.keras.layers.Dense(units=hidden6.shape[1], activation="relu")(hidden6)
outputs = tensorflow.keras.layers.Dense(units=10, activation="softmax")(hidden7)
cnn = tensorflow.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
cnn.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=["accuracy"])
cnn.fit(x=x_train, y=y_train, batch_size=64, epochs=2, validation_split=0.2)
test_scores = cnn.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
y_model = cnn.predict(x=x_test)
print("y_test[1:10]:", numpy.argmax(y_test[1:10], axis=1))
print("y_model[1:10]:", numpy.argmax(y_model[1:10], axis=1))
cnn.summary()
tensorflow.keras.utils.plot_model(cnn, show_shapes=True)

#Classifier image (10 class, image 28x28)
#Create CNN (second variant)
import numpy, tensorflow
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
#x_train.shape=(60000, 28, 28), x_test.shape=(60000, 28, 28)
#y_train.shape=(60000,), y_test.shape=(10000,)
x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
#x_train.shape=(60000, 28, 28, 1), 1 - for grayscale, 3 - for RGB
x_test = x_test.reshape(10000, 28, 28, 1).astype("float32") / 255 
#x_test.shape=(10000, 28, 28, 1), 1 - for grayscale, 3 - for RGB
y_train = tensorflow.keras.utils.to_categorical(y_train)
#y_train.shape=(60000, 10)
y_test = tensorflow.keras.utils.to_categorical(y_test)
#y_test.shape=(60000, 10)
cnn = tensorflow.keras.models.Sequential()
cnn.add(tensorflow.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu")) # filters = convolution planes
cnn.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(tensorflow.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
cnn.add(tensorflow.keras.layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(tensorflow.keras.layers.Flatten()) # transform to 1D layer
cnn.add(tensorflow.keras.layers.Dropout(rate=0.5))
cnn.add(tensorflow.keras.layers.Dense(units=1600, activation="relu"))
cnn.add(tensorflow.keras.layers.Dense(units=10, activation="softmax"))
cnn.compile(optimizer="RMSprop", loss='categorical_crossentropy', metrics=["accuracy"])
cnn.fit(x=x_train, y=y_train, batch_size=64, epochs=2, validation_split=0.2)
test_scores = cnn.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
y_model = cnn.predict_classes(x=x_test)
print("y_test[1:10]:", numpy.argmax(y_test[1:10], axis=1))
print("y_model[1:10]:", y_model[1:10])
cnn.summary()
tensorflow.keras.utils.plot_model(cnn, show_shapes=True)
