from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

def createddataFrame(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label,"completed")
    return image_paths,labels

train = pd.DataFrame()
train['image'], train['label'] = createddataFrame(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createddataFrame(TEST_DIR)

# print(train)

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(48, 48), color_mode='grayscale')
        img = img_to_array(img)
        features.append(img)
    
    features = np.array(features)
    # print(f"Total number of elements in features: {features.size}")
    # print(f"Expected number of elements: {len(features) * 48 * 48}")
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features = extract_features(train['image'])

test_features = extract_features(test['image'])

x_train = train_features/255.0
x_test = test_features/255.0

le = LabelEncoder()
le.fit(train['label'])

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)

model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 5, validation_data = (x_test,y_test)) 

model_json = model.to_json()
with open("driver_model.json",'w') as json_file:
    json_file.write(model_json)
model.save("driver_model.h5")

# ----------------TESTING------------------

json_file = open("driver_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("driver_model.h5")


label = ['angry','disgust','fear','happy','neutral','sad','surprise']
def ef(image):
    img = load_img(image,color_mode= 'grayscale' )
    feature = np.array(img)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0
    
image = 'images/train/angry/43.jpg'
print("original image is of angry")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)


image = 'images/train/happy/7.jpg'
print("original image is of happy")
img = ef(image)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("model prediction is ",pred_label)
plt.imshow(img.reshape(48,48),cmap='gray')