import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import os
import random

# Environment and GPU configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
        )
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
    except RuntimeError as e:
        print(e)

def readucr(filename):
    data = pd.read_excel(filename)
    data = data.values
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def spatial_attention(input_feature):
    kernel_size = 7
    if K.image_data_format() == 'channels_first':
        channel = input_feature.shape[1]
        CRFRI_feature = keras.layers.Permute((2, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        CRFRI_feature = input_feature

    avg_pool = keras.layers.Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(CRFRI_feature)
    max_pool = keras.layers.Lambda(lambda x: K.max(x, axis=2, keepdims=True))(CRFRI_feature)
    concat = keras.layers.Concatenate(axis=2)([avg_pool, max_pool])
    CRFRI_feature = keras.layers.Conv1D(filters=1, kernel_size=kernel_size, activation='hard_sigmoid',
                                        strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)(
        concat)
    if K.image_data_format() == 'channels_first':
        CRFRI_feature = keras.layers.Permute((2, 1))(CRFRI_feature)

    return keras.layers.multiply([input_feature, CRFRI_feature])

def self_attention(input_feature):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    channel = input_feature.shape[channel_axis]

    query = keras.layers.Dense(channel // 4)(input_feature)
    key = keras.layers.Dense(channel // 4)(input_feature)
    value = keras.layers.Dense(channel)(input_feature)

    attention_weights = keras.layers.Dot(axes=(2, 2))([query, key])
    attention_weights = keras.layers.Activation('softmax')(attention_weights)
    output = keras.layers.Dot(axes=(2, 1))([attention_weights, value])

    return keras.layers.Add()([input_feature, output])

def CRFRI_block(CRFRI_feature, ratio=8):
    CRFRI_feature = self_attention(CRFRI_feature)
    CRFRI_feature = spatial_attention(CRFRI_feature)
    return CRFRI_feature

def build_CRFRI(input_shape, n_feature_maps, nb_classes, seed):
    x = keras.layers.Input(shape=(input_shape))

    conv1 = keras.layers.Conv1D(n_feature_maps, 3, 1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                                kernel_initializer=keras.initializers.he_normal(seed=seed))(x)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)
    conv1 = keras.layers.Dropout(0.4, seed=seed)(conv1)  # Further increased dropout rate
    conv1 = keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv1)  # Using max pooling

    CRFRI = CRFRI_block(conv1)
    CRFRI = keras.layers.Add()([conv1, CRFRI])

    conv2 = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                                kernel_initializer=keras.initializers.he_normal(seed=seed))(CRFRI)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv2 = keras.layers.Dropout(0.5, seed=seed)(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv2)

    conv3 = keras.layers.Conv1D(n_feature_maps * 2, 3, 1, padding='same', kernel_regularizer=keras.regularizers.l2(0.01),
                                kernel_initializer=keras.initializers.he_normal(seed=seed))(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv3 = keras.layers.Dropout(0.6, seed=seed)(conv3)
    conv3 = keras.layers.MaxPooling1D(pool_size=2, padding='same')(conv3)

    full = keras.layers.GlobalAveragePooling1D()(conv3)
    out = keras.layers.Dense(nb_classes, activation='softmax', kernel_initializer=keras.initializers.he_normal(seed=seed),
                             kernel_regularizer=keras.regularizers.l2(0.01))(full)
    return x, out

def augment_data(X, y):
    augmented_X = []
    augmented_y = []

    for i in range(len(X)):
        augmented_X.append(X[i])
        augmented_y.append(y[i])

        # Add noise
        noise = np.random.normal(0, 0.05, X[i].shape)
        augmented_X.append(X[i] + noise)
        augmented_y.append(y[i])

        # Scale data
        scale = np.random.uniform(0.8, 1.2)
        augmented_X.append(X[i] * scale)
        augmented_y.append(y[i])

        # Shift data
        shift = np.random.randint(-10, 10)
        augmented_X.append(np.roll(X[i], shift))
        augmented_y.append(y[i])

        # Random flip
        if np.random.rand() > 0.3:
            augmented_X.append(np.flip(X[i], axis=0))
            augmented_y.append(y[i])

    return np.array(augmented_X), np.array(augmented_y)

# Read and preprocess the data
filename = 'C:/Users/cxq/Desktop/dataset_689/1m/1m.xlsx'
x_data, y_data = readucr(filename)
nb_classes = len(np.unique(y_data))

x_data_mean = x_data.mean()
x_data_std = x_data.std()
x_data = (x_data - x_data_mean) / x_data_std
x_data = x_data.reshape(x_data.shape + (1,))

# Convert labels to integer indices
label_mapping = {label: idx for idx, label in enumerate(np.unique(y_data))}
y_data_mapped = np.vectorize(label_mapping.get)(y_data)
Y_data = keras.utils.to_categorical(y_data_mapped, nb_classes)

trainX, testX, trainY, testY = train_test_split(x_data, Y_data, test_size=0.2, random_state=42)

# Data augmentation
trainX_augmented, trainY_augmented = augment_data(trainX, trainY)

batch_size = min(trainX_augmented.shape[0] // 10, 32)  # Increase batch size

inputs = keras.Input(shape=trainX_augmented.shape[1:])
x, y = build_CRFRI(trainX_augmented.shape[1:], 128, nb_classes, seed)
model = keras.models.Model(inputs=x, outputs=y)
optimizer = keras.optimizers.Adam(learning_rate=0.0005)  # Reduced learning rate

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=10, min_lr=0.00001)  # Reduced patience
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

model.fit(trainX_augmented, trainY_augmented, batch_size=batch_size, epochs=100,
              validation_data=(testX, testY),
              callbacks=[reduce_lr, early_stopping], verbose=0)

# Evaluate the model
loss, accuracy, precision, recall = model.evaluate(testX, testY, verbose=0)

# Load the best model
model = keras.models.load_model('C:/Users/cxq/Desktop/dataset_689/1m/model/best_model.h5')

# Evaluating the model
loss, accuracy, precision, recall = model.evaluate(testX, testY, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)

# Predict the test data
y_pred = model.predict(testX)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(testY, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

# Confusion Matrix with customized font size and labels
cm_normalized = cm / cm.sum(axis=1, keepdims=True)  # Normalize and calculate 1 - accuracy

# Custom labels
custom_labels = ['SI', 'MI', 'WF', 'SF', 'MF']
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=custom_labels)
fig, ax = plt.subplots(figsize=(6, 6))  # Set figure size
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical', values_format='.4f')
ax.set_xticks(np.arange(len(custom_labels)))
ax.set_xticklabels(custom_labels, fontsize=16)
ax.set_yticks(np.arange(len(custom_labels)))
ax.set_yticklabels(custom_labels, fontsize=16)

# Remove axis labels
ax.set_xlabel('')
ax.set_ylabel('')

# Adjust font size of internal values
for text in ax.texts:
    text.set_fontsize(16)

# Hide the color bar
ax.images[-1].colorbar.remove()
# plt.title('Confusion Matrix', fontsize=16)

plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16)

plt.show()
