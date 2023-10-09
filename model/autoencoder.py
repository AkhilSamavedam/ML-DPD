import tensorflow as tf
import numpy as np
import os
from config import j

strategy = tf.distribute.MirroredStrategy()


def Encoder(latent_dim, input_shape):
    encoder_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same')(encoder_input)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Flatten()(x)
    encoder_output = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    encoder = tf.keras.Model(encoder_input, encoder_output)
    return encoder

def Decoder(latent_dim, input_shape):
    decoder_input = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(16896, activation='relu')(decoder_input)
    x = tf.keras.layers.Reshape((66, 4, 64))(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2DTranspose(3, (3, 3), activation='selu', padding='same')(x)
    decoder_output = tf.keras.layers.Cropping2D(cropping=((0, 24), (0, 12)))(x)
    return tf.keras.Model(decoder_input, decoder_output)


def load_npy_files(file_paths):
    for file_path in file_paths:
        data = np.load(file_path)
        yield tf.convert_to_tensor(data, dtype=tf.float32)

def create_dataset_from_npy_folder(folder_path=j('Numpy'), batch_size=32, train_val_split=0.8):
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    file_list.sort()

    dataset = tf.data.Dataset.from_generator(
        lambda: load_npy_files([os.path.join(folder_path, file) for file in file_list]),
        output_signature=tf.TensorSpec(shape=None, dtype=tf.float32)  # Variable shape
    )

    # Duplicate the dataset to get both input (x) and target (y) as the same data (for autoencoder)
    dataset = dataset.map(lambda x: (x, x))

    # Split the dataset into training and validation sets
    num_files = len(file_list)
    num_train_files = int(train_val_split * num_files)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]

    # Distribute the datasets using the strategy
    num_files = len(file_list)
    num_train_files = int(train_val_split * num_files)
    train_dataset = dataset.take(num_train_files)
    val_dataset = dataset.skip(num_train_files)

    train_dataset = train_dataset.shuffle(buffer_size=280)
    train_dataset = train_dataset.batch(batch_size * strategy.num_replicas_in_sync)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(batch_size * strategy.num_replicas_in_sync)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


    return train_dataset, val_dataset, file_list

latent_dim = 512 
input_shape = (4200, 244, 3)


# Distribute the datasets with input context using strategy.run
# Define and compile your model here
# ... (rest of the model definition and training code)

# Define the model within the strategy scope
with strategy.scope():

    training_dataset, val_dataset, file_list = create_dataset_from_npy_folder()

    # Define and compile your model here
    encoder = Encoder(latent_dim, input_shape)
    decoder = Decoder(latent_dim, input_shape)

    autoencoder_input = tf.keras.Input(shape=input_shape)
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = tf.keras.Model(autoencoder_input, decoded)

    autoencoder.compile(optimizer='adam', loss='mse')

checkpoint_path = j('results/256/autoencoder_checkpoint.ckpt')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)

num_train_samples = len(file_list) * 0.8
steps_per_epoch = int(np.ceil(num_train_samples / (32 * strategy.num_replicas_in_sync)))


history = autoencoder.fit(
    training_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[checkpoint_callback]  # Include the checkpoint callback during training
)

# Save the training history to a file (CSV format)
history_file = j('results/256/training_history.csv')
with open(history_file, "w") as f:
    f.write("epoch,loss,val_loss\n")
    for epoch, loss, val_loss in zip(range(1, len(history.history['loss']) + 1),
                                     history.history['loss'],
                                     history.history['val_loss']):
        f.write(f"{epoch},{loss},{val_loss}\n")

# Save the trained model (optional)
autoencoder.save(j('results/256/autoencoder_model.h5'))
encoder.save(j('results/256/encoder.h5'))
decoder.save(j('results/256/decoder.h5'))
