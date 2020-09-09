#Classification of a casting product using CNN


#Libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


data = ImageDataGenerator(rescale=1/255.0, validation_split=0.25)

#Training & test Data Reading
trainData = data.flow_from_directory(directory='D:\Bhushan\casting_512x512',
                                           target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 32,
                                           subset='training')

trainData.class_indices

testData = data.flow_from_directory(directory='D:\Bhushan\casting_512x512',
                                           target_size=(128,128),
                                           class_mode = 'binary',
                                           batch_size = 32,
                                           subset='validation')

testData.class_indices

#Deep Learning model
model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(128,128,3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))


model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Model Fitting
history=model.fit_generator(generator=trainData,
                            steps_per_epoch=100,
                            epochs=10,
                            validation_data=testData ,
                            validation_steps=40
                           )

#plot for Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#test Image
test=r'D:\Bhushan\casting_512x512\def_front\cast_def_0_180.jpeg'
img=image.load_img(test,target_size=(128,128))
plt.imshow(img)
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
images=np.vstack([x])

val1=model.predict(images)
if val1==0:
    plt.title("def_front")
else:
    plt.title("ok_front")
    
# Test Data
val=model.predict_classes(testData[0][0])
from sklearn.metrics import confusion_matrix , accuracy_score
p=confusion_matrix(testData[0][1] , val)
acc=accuracy_score(testData[0][1] , val)

