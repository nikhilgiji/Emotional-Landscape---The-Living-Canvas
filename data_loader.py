# data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

def load_data(file_path='../input/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv'):
    df = pd.read_csv(file_path)
    INTERESTED_LABELS = [0, 3, 4]  # Specify your interested labels here
    df = df[df.emotion.isin(INTERESTED_LABELS)]

    img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
    img_array = np.stack(img_array, axis=0)

    le = LabelEncoder()
    img_labels = le.fit_transform(df.emotion)
    img_labels = np_utils.to_categorical(img_labels)

    X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                          shuffle=True, stratify=img_labels,
                                                          test_size=0.1, random_state=42)

    X_train = X_train / 255.
    X_valid = X_valid / 255.

    return X_train, X_valid, y_train, y_valid

def get_label_mapping(le):
    return dict(zip(le.classes_, le.transform(le.classes_)))
