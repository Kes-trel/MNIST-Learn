import pandas as pd
import tensorflow as tf
import streamlit as st
import tensorflow_datasets as tfds
import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
sns.set()

from help_text import *
from help_lists import *

st.set_page_config(page_title="MNIST NN Trainer",page_icon="🔢", layout="wide", initial_sidebar_state="auto")

with st.beta_expander("THE MNIST DATABASE"):
    st.write("The MNIST database of handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting.")
    st.write("For more information see http://yann.lecun.com/exdb/mnist/")

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

def plot_loss_lines():
    fig_line, ax = plt.subplots()
    sns.lineplot(data=df[["Loss","Validation Loss"]]).xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.set(xlabel="Epochs", ylabel="Loss")
    return st.pyplot(fig_line)

def plot_accuracy():
    bar_data = df[["Accuracy %","Validation Accuracy %"]]
    bar_data["Epoch"] = df.index
    bar_data = bar_data.melt(id_vars="Epoch").rename(columns=str.title)
    min_y_axis = round(bar_data.min(axis=0)["Value"]-5,-1)

    fig_bar, ax = plt.subplots()
    sns.barplot(x="Epoch", y="Value", hue="Variable", data=bar_data)
    ax.set(xlabel="Epochs", ylabel="Validation", ylim=(min_y_axis, 100))
    return st.pyplot(fig_bar)

# Streamlit Layout

col_a, col_b = st.beta_columns(2)
with col_a:
    plot_loss_lines()
with col_b:
    plot_accuracy()
    
st.table(df)

with st.beta_expander("TEST THE MODEL"):
    col_1, col_2 = st.beta_columns([2,1])
    col_1.write(test_model_help)
    col_2.image("https://pbs.twimg.com/media/ESY0WNGU4AA3P0S.jpg")
    
    if st.button("Roger that.. run the test"):
        test_loss, test_accuracy = model.evaluate(test_data)
        st.header(f"Test loss is at {test_loss:.3f} and test accuracy is {test_accuracy*100:.1f}%!")