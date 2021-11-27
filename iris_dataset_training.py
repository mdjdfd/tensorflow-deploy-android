import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def training_iris_dataset():
    df = pd.read_csv('/Users/junaidfahad/Downloads/iris.data')
    X = df.iloc[:, :4].values
    y = df.iloc[:, 4].values

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)
    y = tf.keras.utils.to_categorical(y)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=[4]))
    model.add(Dense(64))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

    model.fit(X, y, epochs=400)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tfmodel_convert = converter.convert()

    open('/Users/junaidfahad/Downloads/irismodel.tflite', 'wb').write(tfmodel_convert)
