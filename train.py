import tensorflow as tf
from tf.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf.keras.preprocessing.image import ImageDataGenerator

def train_model(model, X_train, y_train, X_valid, y_valid):
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1,
    )

    callbacks = [early_stopping, lr_scheduler]

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
    )
    train_datagen.fit(X_train)

    batch_size = 32
    epochs = 100

    history = model.fit_generator(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_valid, y_valid),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True
    )

    return history
