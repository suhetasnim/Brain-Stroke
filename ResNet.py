# %%
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
# %%
data_dir = 'Dataset'
train_dir = 'split/train'
test_dir = 'split/test'
val_dir = 'split/val'
# %%
diseases_name = []
for image_class in os.listdir(train_dir):
    diseases_name.append(image_class)
print(diseases_name)
print(f'Total Disease: {len(diseases_name)}')
# %%
total_images = 0
for image_class in os.listdir(data_dir):
    print(image_class+':'+str(len(os.listdir(data_dir+'/'+image_class))))
    total_images += len(os.listdir(data_dir+'/'+image_class))
print(f'Total Images:{total_images}')
# %%
train_data = image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=42
)
test_data = image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
    seed=42
)
val_data = image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
    seed=42
)
# %%
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                              min_delta=0.0001)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2,
                                                 patience=4, min_lr=1e-7)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
# %%
n_class = len(diseases_name)

model = Sequential()

pretrained_model = ResNet50(include_top=False,
                            input_shape=(224, 224, 3),
                            pooling='max', classes=n_class,
                            weights='imagenet')
pretrained_model.trainable = False

pretrained_model = Model(
    inputs=pretrained_model.inputs,
    outputs=pretrained_model.layers[-2].output
)
model.add(pretrained_model)

model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Flatten())

model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(n_class, activation='softmax'))

model.summary()

# %%
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
# %%
hist = model.fit(train_data, epochs=50, validation_data=val_data,
                 callbacks=[
                     early_stop
                 ])
# %%
model.save('saved_models/ResNet50.h5')

# %%
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()
# %%
fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
# %%
test_loss, test_acc = model.evaluate(test_data)
print(f'Test Loss: {test_loss} Test Accuracy: {test_acc}')
# %%
y_pred = model.predict(test_data)
# %%
sns.set()
y_pred = []
for x, y in test_data:
    y_pred.extend(np.argmax(model.predict(x), axis=1).tolist())

y_true = []
for x, y in test_data:
    y_true.extend(np.argmax(y, axis=1).tolist())

conf_mat = confusion_matrix(y_true, y_pred)

sns.heatmap(conf_mat, annot=True, cmap="YlGnBu")
# %%
sns.set()
y_pred = []
for x, y in test_data:
    y_pred.extend(np.argmax(model.predict(x), axis=1).tolist())


y_true = []
for x, y in test_data:
    y_true.extend(np.argmax(y, axis=1).tolist())

conf_mat = confusion_matrix(y_true, y_pred)

sns.heatmap(conf_mat, annot=True, cmap="YlGnBu")
# %%
precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)

print(f'precision: {np.average(precision)}')
print(f'Recall: {np.average(recall)}')
