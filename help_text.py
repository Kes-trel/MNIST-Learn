
buffer_size_help = """
The Buffer Size parameter is here for cases when we're dealing with enormous datasets then we can't shuffle the whole dataset in one go because we can't fit it all in memory so instead TF only stores set size samples in memory at a time and shuffles them.
* Buffer Size = 1 then no shuffling will actually happen.
* Buffer Size >= Number of Samples then shuffling is uniform.
* Buffer Size in between 1 and Number of Samples then a computational optimization to approximate uniform shuffling.
"""

batch_size_help = """
A batch size of 1 results in the Stochastic Gradient Descent (SGD). It takes the algorithm very little time to process a single batch (as it is one data point), but there are thousands of batches (54,000 in MNIST), thus the algorithm is actually slow. The middle ground (mini-batching such as 100 samples per batch) is optimal.

* Notice that the validation accuracy starts from a high number. That's because there are lots updates in a single epoch. Once the training is over, the accuracy is lower than all other batch sizes (SGD was an approximation).

A bigger batch size results in slower training. We are taking advantage of batching because of the speed increase.

* Notice that the validation accuracy starts from a low number and with 5 epochs actually finishes at a lower number. That's because there are fewer updates in a single epoch. Try a batch size of 30,000 or 50,000. That's very close to single batch Gradient Descent (GD) for this problem. What do you think about the speed? You will need to change the max epochs to 100 (for instance), as 5 epochs won't be enough to train the model. What do you think about the speed of optimization?
"""

hidden_layer_number_help = """
In neural networks, a hidden layer is located between the input and output of the algorithm, in which the function applies weights to the inputs and directs them through an activation function as the output. In short, the hidden layers perform nonlinear transformations of the inputs entered into the network. Hidden layers vary depending on the function of the neural network, and similarly, the layers may vary depending on their associated weights.
"""

hidden_layer_help = """
To keep simple here we will use same hidden layer unit size for all hidden layers. If you think this is too simple you should not be using this site my friend.
* Try 200 or 500 and so on... The validation accuracy is significantly higher (as the algorithm with 50 hidden units was too simple of a model). Naturally, it takes the algorithm much longer to train (unless early stopping is triggered too soon).
"""

activation_function_help = """
Adjust the activations from 'relu' to 'sigmoid'.

Generally, we should reach an inferior solution. That is because relu 'cleans' the noise in the data (think about it - if a value is negative, relu filters it out, while if it is positive, it takes it into account). For the MNIST dataset, we care only about the intensely black and white parts in the images of the digits, so such filtering proves beneficial.

The sigmoid does not filter the signals as well as relu, but still reaches a respectable result (around 95%).

For Tensor Flow - built-in activation functions see:
https://www.tensorflow.org/api_docs/python/tf/keras/activations
"""

model_optimizer_help = """
Built-in optimizer classes.

* Adadelta: Optimizer that implements the Adadelta algorithm.
* Adagrad: Optimizer that implements the Adagrad algorithm.
* Adam: Optimizer that implements the Adam algorithm.
* Adamax: Optimizer that implements the Adamax algorithm.
* Ftrl: Optimizer that implements the FTRL algorithm.
* Nadam: Optimizer that implements the NAdam algorithm.
* Optimizer: Base class for Keras optimizers.
* RMSprop: Optimizer that implements the RMSprop algorithm.
* SGD: Gradient descent (with momentum) optimizer.

https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
"""

model_loss_help = """
Functions
* KLD(...): Computes Kullback-Leibler divergence loss between y_true and y_pred.
* MAE(...): Computes the mean absolute error between labels and predictions.
* MAPE(...): Computes the mean absolute percentage error between y_true and y_pred.
* MSE(...): Computes the mean squared error between labels and predictions.
* MSLE(...): Computes the mean squared logarithmic error between y_true and y_pred.
* binary_crossentropy(...): Computes the binary crossentropy loss.
* categorical_crossentropy(...): Computes the categorical crossentropy loss.
* categorical_hinge(...): Computes the categorical hinge loss between y_true and y_pred.
* cosine_similarity(...): Computes the cosine similarity between labels and predictions.
* deserialize(...): Deserializes a serialized loss class/function instance.
* get(...): Retrieves a Keras loss as a function/Loss class instance.
* hinge(...): Computes the hinge loss between y_true and y_pred.
* huber(...): Computes Huber loss value.
* kl_divergence(...): Computes Kullback-Leibler divergence loss between y_true and y_pred.
* kld(...): Computes Kullback-Leibler divergence loss between y_true and y_pred.
* kullback_leibler_divergence(...): Computes Kullback-Leibler divergence loss between y_true and y_pred.
* log_cosh(...): Logarithm of the hyperbolic cosine of the prediction error.
* logcosh(...): Logarithm of the hyperbolic cosine of the prediction error.
* mae(...): Computes the mean absolute error between labels and predictions.
* mape(...): Computes the mean absolute percentage error between y_true and y_pred.
* mean_absolute_error(...): Computes the mean absolute error between labels and predictions.
* mean_absolute_percentage_error(...): Computes the mean absolute percentage error between y_true and y_pred.
* mean_squared_error(...): Computes the mean squared error between labels and predictions.
* mean_squared_logarithmic_error(...): Computes the mean squared logarithmic error between y_true and y_pred.
* mse(...): Computes the mean squared error between labels and predictions.
* msle(...): Computes the mean squared logarithmic error between y_true and y_pred.
* poisson(...): Computes the Poisson loss between y_true and y_pred.
* serialize(...): Serializes loss function or Loss instance.
* sparse_categorical_crossentropy(...): Computes the sparse categorical crossentropy loss.
* squared_hinge(...): Computes the squared hinge loss between y_true and y_pred.
"""

epochs_help = """
An epoch indicates the number of passes of the entire training dataset the machine learning algorithm has completed. Determining how many epochs a model should run to train is based on many parameters related to both the data itself and the goal of the model.
"""