import cv2
import numpy as np
import csv
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers import Lambda, Cropping2D, Dropout, ELU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_csv(file):
    lines = []
    with open(file) as csvfile:
        load = csv.reader(csvfile)
        for line in load:
            lines.append(line)
    return lines[1:]


def load_image(lines_path, image_path):
    images = []
    angles = []
    for line in lines_path:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = image_path + filename
            image = mpimg.imread(current_path)
            images.append(image)
            angle = float(line[3])
            if i == 0:
                angles.append(angle)
            elif i == 1:
                angles.append(angle + 0.20)
            else:
                angles.append(angle - 0.20)
    X_train = np.array(images)
    y_train = np.array(angles)
    shuffle(X_train, y_train)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    return X_train, X_validation, y_train, y_validation


def image_generator(X, y):
    X_w_generator = []
    y_w_generator = []
    for X_generate, y_generate in zip(X, y):
        X_w_generator.append(X_generate)
        y_w_generator.append(y_generate)
        extra_X = cv2.flip(X_generate, 1)
        extra_y = y_generate * (-1.0)
        X_w_generator.append(extra_X)
        y_w_generator.append(extra_y)
    X_total = np.array(X_w_generator)
    y_total = np.array(y_w_generator)
    return shuffle(X_total, y_total)


def resize_img(img):
    from keras.backend import tf as ktf
    resize = ktf.image.resize_images(img, (60, 120))
    resize = resize / 255.0 - 0.5
    return resize


def show_steering(y_train, y_valid):
    max_degree = 25
    degree_per_steering =10
    n_classes = max_degree * degree_per_steering
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    plt.subplots_adjust(left=0, right=0.95, top=0.9, bottom=0.25)
    ax0, ax1= axes.flatten()
    ax0.hist(y_train, bins=n_classes, histtype='bar', color='blue', rwidth=0.6, label='train')
    ax0.set_title('Number of training')
    ax0.set_xlabel('Steering Angle')
    ax0.set_ylabel('Total Image')
    ax1.hist(y_valid, bins=n_classes, histtype='bar', color='red', rwidth=0.6, label='valid')
    ax1.set_title('Number of validation')
    ax1.set_xlabel('Steering Angle')
    ax1.set_ylabel('Total Image')
    fig.tight_layout()
    plt.show()
    
# Load csv file and images
file = 'D:/97_DL/VS-Behavior-Cloning/Behavior-Cloning/Behavior-Cloning/data/data_1/driving_log.csv'
image_path = 'D:/97_DL/VS-Behavior-Cloning/Behavior-Cloning/Behavior-Cloning/data/data_1/IMG/'
lines_path = read_csv(file)
# Original traning examples and labels
X_train, X_validation, y_train, y_validation = load_image(lines_path, image_path)
assert len(X_train) == len(y_train), "ERROR: Training example size error".format(len(X_train), len(y_train))
assert len(X_validation) == len(y_validation), "ERROR: Validation example size error".format(len(X_validation), len(y_validation))
print("Training examples: {}\nValidation examples: {}".format(len(X_train), len(X_validation)))
show_steering(y_train, y_validation)
# After image generator
X_train_gen, y_train_gen = image_generator(X_train, y_train)
X_valid_gen, y_valid_gen = image_generator(X_validation, y_validation)
assert len(X_train_gen) == len(y_train_gen), "ERROR: Total training example size error".format(len(X_train_gen), len(y_train_gen))
assert len(X_valid_gen) == len(y_valid_gen), "ERROR: Total validation example size error".format(len(X_valid_gen), len(y_valid_gen))
print("Total training examples: {}\nTotal validation examples: {}".format(len(X_train_gen), len(X_valid_gen)))
show_steering(y_train_gen, y_valid_gen)

# Model
model = Sequential()
model.add(Cropping2D(cropping=((75,25),(0,0)), input_shape=(X_train_gen[0].shape[0], X_train_gen[0].shape[1], X_train_gen[0].shape[2]), data_format="channels_last"))
model.add(Lambda(resize_img, input_shape=(160,320,3), output_shape=(60,120,3)))

model.add(Conv2D(3, (1,1), padding='same'))
model.add(ELU())
model.add(BatchNormalization())

model.add(Conv2D(16, (5,5), strides=(2,2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())

model.add(Conv2D(32, (5,5), strides=(2,2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())

model.add(Conv2D(64, (3,3), strides=(2,2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())

model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
model.add(ELU())
model.add(BatchNormalization())

model.add(Flatten())
model.add(ELU())

model.add(Dense(512))
model.add(Dropout(0.2))
model.add(ELU())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(ELU())

model.add(Dense(10))
model.add(Dropout(0.5))
model.add(ELU())

model.add(Dense(1))

adam = Adam(lr=1e-5)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

model.summary()

checkpointer = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
results = model.fit(X_train_gen, y_train_gen, validation_data=(X_valid_gen, y_valid_gen),batch_size=32, epochs=50, verbose =1)
model.save('D:/97_DL/VS-Behavior-Cloning/Behavior-Cloning/Behavior-Cloning/model.h5')