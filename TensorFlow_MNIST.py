import pandas as pd
import tensorflow as tf
import streamlit as st
import tensorflow_datasets as tfds
import time
import altair as alt
from help_text import *
from help_lists import *

#inputs from streamlit app

BUFFER_SIZE = st.sidebar.number_input("Buffer Size", value=100, min_value=1, step=100, format="%i", help=buffer_size_help)
BATCH_SIZE = st.sidebar.number_input("Batch Size", value=100, min_value=1, step=100, format="%i", help=batch_size_help)
hidden_layers_number = st.sidebar.number_input("Number of Hidden Layers", value=2, min_value=1, step=1, format="%i", help=hidden_layer_number_help)
hidden_layer_size = st.sidebar.number_input("Hidden Layer Unit Size", value=50, min_value=1, step=10, format="%i", help=hidden_layer_help)
activation_function = st.sidebar.selectbox("Select Activation Function", activation_functions_list, index=7, help=activation_function_help)
model_optimizer = st.sidebar.selectbox("Select Model Optimiser Class", model_optimizer_list, index=2, help=model_optimizer_help)
model_loss = st.sidebar.selectbox("Select Model Loss", model_loss_list, index=28, help=model_loss_help)
NUM_EPOCHS = st.sidebar.number_input("Number of epochs", value=5, min_value=1, step=1, format="%i", help=epochs_help)


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

shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(BUFFER_SIZE)
validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
train_data = shuffled_train_and_validation_data.skip(num_validation_samples)

train_data = train_data.batch(BATCH_SIZE)
validation_data = validation_data.batch(num_validation_samples)
test_data = test_data.batch(num_test_samples)
validation_inputs, validation_targets = next(iter(validation_data))

# MODEL
input_size = 784    # 28x28x1 pixels, therefore it is a tensor of rank 3 - just remember to flatten it to a single vector
output_size = 10    # 0 to 9

# Use same hidden layer size for both hidden layers

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1))) # input layer
for layer in range(hidden_layers_number):
    model.add(tf.keras.layers.Dense(hidden_layer_size, activation=activation_function)) # hidden layers
model.add(tf.keras.layers.Dense(output_size, activation='softmax')) # output layer

model.compile(optimizer=model_optimizer, loss=model_loss, metrics=["accuracy"])

# Custom callback for epoch timings just like you get from model fit verbose

class EpochTimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time()-self.starttime)

epoch_time = EpochTimingCallback()

# Train the model
history = model.fit(train_data, epochs=NUM_EPOCHS, validation_data=(validation_inputs, validation_targets), verbose =1, callbacks=[epoch_time])


# Streamlit section

df = pd.DataFrame.from_dict(history.history)
df.columns = ["Loss", "Accuracy %", "Validation Loss", "Validation Accuracy %"]
df["Accuracy %"] = df["Accuracy %"] * 100
df["Validation Accuracy %"] = df["Validation Accuracy %"] * 100
df.insert(loc=0, column="Epoch Time(s)", value=epoch_time.logs)
df.index += 1


# def change_bar_chart(dataframe_in):
#     price_bar_chart = alt.Chart(dataframe_in).mark_bar().encode(
#                     x=alt.X(dataframe_in.index, title=None),
#                     y=alt.Y("Loss", title="Percentage in "),
#                     tooltip=["Loss", "Accuracy %"]
#                 ).properties(height=500)
#     return st.altair_chart(price_bar_chart, use_container_width=True)

# change_bar_chart(df)

st.table(df)
