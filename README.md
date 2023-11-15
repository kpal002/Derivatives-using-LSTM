# Derivatives-using-LSTM
Approach

Problem Statement
The goal is to create an LSTM-based model that can accurately predict the derivative of a given mathematical function.

Data Processing
- Tokenization: Mathematical expressions are tokenized into numbers, variables, operators, and functions.
- Vectorization: Tokenized expressions are vectorized using a built vocabulary, transforming them into numerical formats suitable for model processing.
- Padding: Sequences are padded to a fixed length to ensure consistent input size for the LSTM model.

Model Description
The core of the project is the DerivativeLSTM class, a bidirectional LSTM model designed to process sequences of tokenized mathematical expressions and predict their derivatives.

Training Process
The model is trained on a dataset of mathematical functions and their corresponding derivatives. There are a total of million samples in the dataset. Training involves minimizing the loss between the predicted and actual derivatives, with a focus on accuracy as the primary performance metric.

Key Parameters and Dimensions

- vocab_size: The size of the vocabulary for the input functions. (69)
- embedding_dim: The dimensionality of the embedding space. (256)
- hidden_size: The number of features in the hidden state of each LSTM layer. (256)
- output_size: The size of the vocabulary for the output derivatives. (69)
- num_layers: The number of LSTM layers in the model. (2)

Training involves minimizing the loss between the predicted and actual derivatives, with a focus on accuracy as the primary performance metric. The entire dataset was split into train:val:test in the ratio 75:10:15. Mean accuracy on the holdout test data is 0.84. However, if instead of the entire expression of the predicted derivative matching with the true derivative, we focus on character level matching, the mean accuracy score goes upto 0.97.
![Unknown-6](https://github.com/kpal002/Derivatives-using-LSTM/assets/49849134/19d25314-b37a-4941-baa1-2c87513e5b06)


