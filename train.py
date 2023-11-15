# -*- coding: utf-8 -*-
import re
import torch
import string
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from collections import Counter

# Function to tokenize mathematical expressions
def tokenize_expression(expression):
    """
    Tokenize a mathematical expression into numbers, variables, operators, and functions.

    Args:
    expression (str): The mathematical expression to tokenize.

    Returns:
    list: A list of tokens extracted from the expression.
    """
    tokens = re.findall(r'\b\w+|\W+', expression)
    return tokens

# Function to parse dataset from a file
def parse_dataset(file_path):
    """
    Parse a dataset from a given file path. Assumes each line contains a function and its derivative.

    Args:
    file_path (str): Path to the dataset file.

    Returns:
    list: A list of dictionaries, each containing a function, its variable, and derivative.
    """
    parsed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                function, derivative = line.strip().split('=')
                variable = re.search(r'd\((.*?)\)/d(\w+)', function).group(2)
                function = re.search(r'd\((.*?)\)/d\w+', function).group(1)
                parsed_data.append({
                    'function': tokenize_expression(function),
                    'variable': variable,
                    'derivative': tokenize_expression(derivative)
                })
    return parsed_data


# Function to build a vocabulary from a list of texts
def build_vocab(texts, extra_chars=None):
    """
    Build a vocabulary that includes all characters in the given texts
    and additional specified characters, if provided.

    Args:
    texts (list of str): List of text samples.
    extra_chars (list of str, optional): List of additional characters to include in the vocabulary.

    Returns:
    dict: Mapping of each character to a unique integer.
    """
    counter = Counter()
    for text in texts:
        counter.update(text)

    if extra_chars:
        for char in extra_chars:
            counter[char] = counter.get(char, 0)

    return {char: i+1 for i, char in enumerate(sorted(counter))}  # +1 for zero padding


def get_func_vocab(parsed_dataset, extra_chars=None):
    """
    Generate the vocabulary for functions from the parsed dataset.

    This function iterates over the parsed dataset to extract function expressions,
    then builds a vocabulary that includes all unique characters in these expressions.
    Additional characters can be included in the vocabulary if specified.

    Args:
    parsed_dataset (list of dict): Parsed dataset where each entry contains function and derivative expressions.
    extra_chars (list of str, optional): Additional characters to include in the vocabulary.

    Returns:
    dict: A dictionary mapping each unique character in the function expressions to a unique integer.
    """
    functions = ["".join(data['function']) for data in parsed_dataset]
    return build_vocab(functions, extra_chars=extra_chars)

def get_deriv_vocab(parsed_dataset, extra_chars=None):
    """
    Generate the vocabulary for derivatives from the parsed dataset.

    Similar to get_func_vocab, this function builds a vocabulary from the derivative expressions
    in the parsed dataset. Additional characters can be included if specified.

    Args:
    parsed_dataset (list of dict): Parsed dataset where each entry contains function and derivative expressions.
    extra_chars (list of str, optional): Additional characters to include in the vocabulary.

    Returns:
    dict: A dictionary mapping each unique character in the derivative expressions to a unique integer.
    """
    derivatives = ["".join(data['derivative']) for data in parsed_dataset]
    return build_vocab(derivatives, extra_chars=extra_chars)


# Vectorization of the functions and derivatives
def vectorize(texts, vocab):
    """
    Vectorize a list of texts based on a given vocabulary.

    Args:
    texts (list): A list of strings to vectorize.
    vocab (dict): A dictionary mapping characters to integers.

    Returns:
    list: A list of torch tensors representing the vectorized texts.
    """
    return [torch.tensor([vocab[char] for char in text]) for text in texts]


# Definition of the DerivativeLSTM model class
class DerivativeLSTM(nn.Module):
    """
    A bidirectional LSTM model for calculating derivatives of functions.

    Args:
    vocab_size (int): Size of the vocabulary.
    embedding_dim (int): Dimension of the embedding layer.
    hidden_size (int): Number of features in the hidden state of the LSTM.
    output_size (int): Size of the output layer.
    num_layers (int): Number of layers in the LSTM.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers):
        """
        Initialize the layers of the DerivativeLSTM model.
        """
        super(DerivativeLSTM, self).__init__()

        # Store the size of hidden layers and number of layers for later use
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Embedding layer to convert token indices into embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional LSTM for processing sequences in both directions
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)

        # Fully connected layer that maps LSTM outputs to the output size
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiplied by 2 for bidirectional

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
        x (Tensor): Input tensor containing sequences of token indices.

        Returns:
        Tensor: Output tensor containing the model predictions.
        """
        # Apply the embedding layer to the input
        embedded = self.embedding(x)

        # Forward propagate through the bidirectional LSTM
        out, _ = self.lstm(embedded)

        # Reshape the output from the LSTM for the fully connected layer
        out = out.contiguous().view(-1, self.hidden_size * 2)

        # Pass through the fully connected layer to get the final output
        out = self.fc(out)

        # Reshape the output to (batch_size, seq_length, output_size)
        out = out.view(x.size(0), -1, self.output_size)

        return out


def manual_model_summary(model):
    """
    Print a summary of a PyTorch model including layer names,
    number of parameters in each layer, and whether they are trainable.

    Args:
    model (torch.nn.Module): The PyTorch model to summarize.
    """

    print("Model Summary:")
    # Format the header of the summary table
    header_format = "{:<20} {:<15} {:<10}"
    print(header_format.format("Layer", "Parameters", "Trainable"))
    print("-" * 45)

    total_params = 0
    total_trainable_params = 0

    # Iterate through all named parameters in the model
    for name, parameter in model.named_parameters():
        # Count the number of parameters in each layer
        param_count = parameter.numel()
        # Check if the parameter is trainable
        trainable = "True" if parameter.requires_grad else "False"
        # Accumulate the total and trainable parameter counts
        total_params += param_count
        if parameter.requires_grad:
            total_trainable_params += param_count
        
        # Print the summary for each layer
        print("{:<20} {:<15} {:<10}".format(name, param_count, trainable))

    # Print the footer with total parameter counts
    print("-" * 45)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {total_trainable_params:,}")


def calculate_test_accuracy(model, test_loader, device):
    """
    Calculate the accuracy of the model on the test dataset.

    Args:
    model (nn.Module): Trained PyTorch model.
    test_loader (DataLoader): DataLoader for the test dataset.
    device (torch.device): Device to run the model on ('cpu' or 'cuda').

    Returns:
    float: Accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for functions, derivatives in test_loader:
            functions, derivatives = functions.to(device), derivatives.to(device)
            outputs = model(functions)
            _, predicted = torch.max(outputs, 2)  # Get the class with the highest probability

            total += derivatives.numel()
            correct += (predicted == derivatives).sum().item()

    accuracy = correct / total
    return accuracy

def display_model_predictions(model, test_loader, deriv_vocab, num_samples=5):
    # Ensure model is in evaluation mode
    model.eval()

    # Reverse the derivative vocabulary (index to token)
    reverse_deriv_vocab = {index: char for char, index in deriv_vocab.items()}

    # Collect all samples from the test set
    all_samples = [(func, deriv) for func, deriv in test_loader.dataset]

    # Select random samples
    random_samples = random.sample(all_samples, num_samples)

    with torch.no_grad():  # Disable gradient computation
        for i, (func, deriv) in enumerate(random_samples, start=1):
            func = func.to(device).unsqueeze(0)  # Add batch dimension
            outputs = model(func)
            _, predicted = torch.max(outputs, 2)  # Get the class with the highest probability

            # Convert predicted and actual token indices back to characters, skipping padding tokens
            predicted_deriv = ''.join([reverse_deriv_vocab.get(index.item(), '') for index in predicted[0] if index.item() != 0])
            actual_deriv = ''.join([reverse_deriv_vocab.get(index.item(), '') for index in deriv if index.item() != 0])

            print(f"Sample {i} Predicted Derivative: {predicted_deriv}")
            print(f"Sample {i} Actual Derivative: {actual_deriv}\n")



if __name__ == "__main__":
    # Example usage of the parse_dataset function
    file_path = 'train.txt'
    parsed_dataset = parse_dataset(file_path)
    
    # Extracting functions and derivatives
    functions = ["".join(data['function']) for data in parsed_dataset]
    derivatives = ["".join(data['derivative']) for data in parsed_dataset]
    
    # Specify additional characters (e.g., all alphabets)
    extra_chars = list(string.ascii_letters)

    # Build vocabularies for functions and derivatives
    func_vocab = build_vocab(functions, extra_chars=extra_chars)
    deriv_vocab = build_vocab(derivatives, extra_chars=extra_chars)
    
    # Preprocessing and splitting dataset
    functions = ["".join(data['function']) for data in parsed_dataset]
    derivatives = ["".join(data['derivative']) for data in parsed_dataset]
    func_vocab = build_vocab(functions)
    deriv_vocab = build_vocab(derivatives)
    func_vectorized = vectorize(functions, func_vocab)
    deriv_vectorized = vectorize(derivatives, deriv_vocab)

    # Padding sequences to a maximum length and creating tensor datasets
    max_length = 30
    func_padded = pad_sequence(func_vectorized, batch_first=True, padding_value=0)
    func_padded = torch.nn.functional.pad(func_padded, (0, max(0, max_length - func_padded.shape[1])), value=0)[:,:max_length]
    deriv_padded = pad_sequence(deriv_vectorized, batch_first=True, padding_value=0)
    deriv_padded = torch.nn.functional.pad(deriv_padded, (0, max(0, max_length - deriv_padded.shape[1])), value=0)[:,:max_length]

    # Splitting the dataset into training, validation, and test sets
    func_train_val, func_test, deriv_train_val, deriv_test = train_test_split(func_padded, deriv_padded, test_size=0.15)
    func_train, func_val, deriv_train, deriv_val = train_test_split(func_train_val, deriv_train_val, test_size=0.1)

    # Creating TensorDatasets for each set
    train_dataset = TensorDataset(func_train, deriv_train)
    val_dataset = TensorDataset(func_val, deriv_val)
    test_dataset = TensorDataset(func_test, deriv_test)

    # DataLoader for batch processing
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    
    # Device configuration and model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    vocab_size = len(func_vocab) + 1
    embedding_dim = 256
    hidden_size = 256
    output_size = len(deriv_vocab) + 1
    num_layers = 2
    model = DerivativeLSTM(vocab_size, embedding_dim, hidden_size, output_size, num_layers).to(device)
    manual_model_summary(model)

    # Training configuration
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_epochs = 10

    # Training and validation loop
    best_val_accuracy = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Initialize metrics for training
        total_train_loss = 0
        total_train_correct = 0
        total_train_tokens = 0

        # Initialize metrics for validation
        total_val_loss = 0
        total_val_correct = 0
        total_val_tokens = 0

        # Set the model to training mode
        model.train()

        # Training loop
        for functions, derivatives in train_loader:
            # Move data to the appropriate device (GPU or CPU)
            functions, derivatives = functions.to(device), derivatives.to(device)

            # Forward pass: compute the model output
            outputs = model(functions)
            outputs = outputs.view(-1, outputs.shape[2])  # Reshape for loss calculation
            loss = criterion(outputs, derivatives.view(-1))

            # Backward pass: compute the gradient and step optimizer
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()        # Compute gradients of all variables wrt loss
            optimizer.step()       # Perform updates using calculated gradients

            # Accumulate the training loss and accuracy
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_correct += (predicted == derivatives.view(-1)).sum().item()
            total_train_tokens += derivatives.numel()

        # Calculate the average loss and accuracy for this epoch
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = total_train_correct / total_train_tokens

        # Append training metrics to lists for plotting
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Set the model to evaluation mode
        model.eval()
    
        with torch.no_grad():  # Disable gradient computation
            # Validation loop
            for func_val, deriv_val in val_loader:
                # Move data to the appropriate device
                func_val, deriv_val = func_val.to(device), deriv_val.to(device)

                # Compute model output
                outputs_val = model(func_val)
                outputs_val = outputs_val.view(-1, outputs_val.shape[2])
                loss_val = criterion(outputs_val, deriv_val.view(-1))

                # Accumulate the validation loss and accuracy
                total_val_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val, 1)
                total_val_correct += (predicted_val == deriv_val.view(-1)).sum().item()
                total_val_tokens += deriv_val.numel()

            # Calculate the average loss and accuracy for validation
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = total_val_correct / total_val_tokens

            # Append validation metrics to lists for plotting
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)

            # Save the model if the validation accuracy is the best so far
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Saved new best model at epoch {epoch+1}")

        # Print metrics for the epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


    # Load the trained model
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)

    # Calculate the test accuracy
    test_accuracy = calculate_test_accuracy(model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # See some of the results.
    display_model_predictions(model, test_loader, deriv_vocab)

    # Set the style of seaborn
    sns.set(style="whitegrid")

    # Plotting training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    
    
    

