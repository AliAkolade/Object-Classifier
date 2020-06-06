from keras.datasets import cifar100

labels = ['beaver, dolphin, otter, seal, whale, aquarium fish, flatfish, ray, shark, trout, orchids, poppies,'
          ' roses, sunflowers, tulips, bottles, bowls, cans, cups, plates, apples, mushrooms, oranges, pears,'
          ' sweet peppers, clock, computer keyboard, lamp, telephone, television, bed, chair, couch, table,'
          ' wardrobe, bee, beetle, butterfly, caterpillar, cockroach,bear, leopard, lion, tiger, wolf, bridge,'
          ' castle, house, road, skyscraper, cloud, forest, mountain, plain, sea, camel, cattle, chimpanzee,'
          ' elephant, kangaroo, fox, porcupine, possum, raccoon, skunk, crab, lobster, snail, spider, worm,'
          ' baby, boy, girl, man, woman, crocodile, dinosaur, lizard, snake, turtle, hamster, mouse, rabbit,'
          ' shrew, squirrel, maple, oak, palm, pine, willow, bicycle, bus, motorcycle, pickup truck, train,'
          ' lawn-mower, rocket, streetcar, tank, tractor']

(X_train, y_train), (X_test, y_test) = cifar100.load_data()

from keras.utils import np_utils

new_X_train = X_train.astype('float32')
new_X_test = X_test.astype('float32')
new_X_train /= 255
new_X_test /= 255
new_Y_train = np_utils.to_categorical(y_train)
new_Y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
model.fit(new_X_train, new_Y_train, epochs=100, batch_size=32)

model.save('100_Trained_model.h5')
