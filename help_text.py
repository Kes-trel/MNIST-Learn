
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

hidden_layer_help = """

"""