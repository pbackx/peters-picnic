from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

HEIGHT = 224
WIDTH = 224

#split in training and validation data https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#yay, also works on dataframes: https://stackoverflow.com/a/24151789/227081

train, valid = train_test_split(pd.read_csv('train.tsv', sep='\t', header=0), test_size=0.2)
print(train.size)
print(train.head())
print(valid.size)
print(valid.head())

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(HEIGHT, WIDTH))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(prefix, img_paths):
    list_of_tensors = [path_to_tensor(prefix + img_path) for img_path in tqdm(img_paths)]
    return preprocess_input(np.vstack(list_of_tensors))

X_train = paths_to_tensor("train/", np.array(train['file']))
X_valid = paths_to_tensor("train/", np.array(valid['file']))

labels = sorted(train['label'].unique())

one_hot_encoding = preprocessing.LabelBinarizer()
one_hot_encoding.fit(labels)
y_train = one_hot_encoding.transform(np.array(train['label']))
y_valid = one_hot_encoding.transform(np.array(valid['label']))

model = Sequential()
model.add(InceptionResNetV2(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3)))
for layer in model.layers:
    layer.trainable = False

model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(labels), activation='softmax'))
print(model.summary())

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionResNetV2.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(X_train, y_train, 
          validation_data=(X_valid, y_valid),
          epochs=200, batch_size=20, callbacks=[checkpointer], verbose=1)
