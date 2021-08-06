import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# import data
# train data
npz = np.load('Data_train.npz')
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)
# validation data
npz = np.load('Data_validation.npz')
validation_inputs = npz['inputs'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)
# test data
npz = np.load('Data_test.npz')
test_inputs = npz['inputs'].astype(np.float)
test_targets = npz['targets'].astype(np.int)

# model

output_size = 1
hidden_layer_size = 7
early_stooping = tf.keras.callbacks.EarlyStopping(patience=2)
# no_hidden_layers = 2
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(
                                hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(
                                hidden_layer_size, activation='relu'),
                            tf.keras.layers.Dense(
                                output_size, activation='sigmoid')

                            ])
custom_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001)   # optimizer settings
model.compile(optimizer=custom_optimizer,
              loss='binary_crossentropy', metrics=['accuracy'])
# prepere batch

BATCH_SIZE = 32
NUM_EPOCHS = 100

# training

model.fit(train_inputs,
          train_targets,
          batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS,
          callbacks=[early_stooping],
          validation_data=(validation_inputs, validation_targets),
          verbose=2
          )


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)

print('Final test accuracy: ', test_accuracy)

yPredict = model.predict(test_inputs)
yPredict = np.where(yPredict > 0.5, 1, 0)

cm=confusion_matrix(test_targets,yPredict)
print(cm)
