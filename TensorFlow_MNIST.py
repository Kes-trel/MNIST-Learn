import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import tensorflow_datasets as tfds


#inputs from streamlit app

buffer_size_help = """
The Buffer Size parameter is here for cases when we're dealing with enormous datasets then we can't shuffle the whole dataset in one go because we can't fit it all in memory so instead TF only stores set size samples in memory at a time and shuffles them.
* Buffer Size = 1 then no shuffling will actually happen.
* Buffer Size >= Number of Samples then shuffling is uniform.
* Buffer Size in between 1 and Number of Samples then a computational optimization to approximate uniform shuffling.
"""

BUFFER_SIZE = st.sidebar.number_input("Buffer Size", value=1000, min_value=1, step=1, format="%i" , help=buffer_size_help)
# BUFFER_SIZE = 1000
# BATCH_SIZE = 100
# hidden_layer_size = 50
# activation_function = "relu"
# model_optimizer = "adam"
# model_loss = "sparse_categorical_crossentropy"
# NUM_EPOCHS = int(st.sidebar.number_input("Number of epochs", min_value=1, step=1, help="Don't go crazy here"))

st.write(type(BUFFER_SIZE))

# as_supervised=True will load the dataset in a 2-tuple structure (input, target) 
mnist_dataset, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)

mnist_train, mnist_test = mnist_dataset['train'], mnist_dataset['test']

# by default, TF has training and testing datasets, but no validation sets - define validation sets
num_validation_samples = 0.1 * mnist_info.splits['train'].num_examples

num_validation_samples = tf.cast(num_validation_samples, tf.int64)

num_test_samples = mnist_info.splits['test'].num_examples

num_test_samples = tf.cast(num_test_samples, tf.int64)

def scale(image, label):
    image = tf.cast(image, tf.float32)
    # inputs are 0 to 255, divide each element by 255 and all elements will be between 0 and 1 
    image /= 255.

    return image, label

scaled_train_and_validation_data = mnist_train.map(scale)
test_data = mnist_test.map(scale)

# BUFFER_SIZE = 1000

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

# BATCH_SIZE = 100

train_data = train_data.batch(BATCH_SIZE)

validation_data = validation_data.batch(num_validation_samples)

test_data = test_data.batch(num_test_samples)

validation_inputs, validation_targets = next(iter(validation_data))

# MODEL

input_size = 784    # 28x28x1 pixels, therefore it is a tensor of rank 3 - just remember to flatten it to a single vector
output_size = 10    # 0 to 9

# Use same hidden layer size for both hidden layers
# hidden_layer_size = 50
# activation_function = "relu"

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # input layer
    
    # tf.keras.layers.Dense => activation(dot(input, weight) + bias)
    tf.keras.layers.Dense(hidden_layer_size, activation=activation_function), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation=activation_function), # 2nd hidden layer
    
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

# model_optimizer = "adam"
# model_loss = "sparse_categorical_crossentropy"

model.compile(optimizer=model_optimizer, loss=model_loss, metrics=["accuracy"])

# TRAIN THE MODEL

# NUM_EPOCHS = 5

history = model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =0)

df = pd.DataFrame.from_dict(history.history)
df.columns = ["Loss", "Accuracy %", "Validation Loss", "Validation Accuracy %"]
df["Accuracy %"] = df["Accuracy %"] * 100
df["Validation Accuracy %"] = df["Validation Accuracy %"] * 100
df.insert(loc=0, column="Epoch", value=df.index+1)

st.write(history.history)

st.table(df)

