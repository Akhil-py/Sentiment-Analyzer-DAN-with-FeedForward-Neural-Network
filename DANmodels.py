import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset


class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        vectors = []
        for sentence in self.sentences:
            # Split sentence into words and get their embeddings
            sentence_vectors = []
            for word in sentence.split():  # Split sentence into words
                vector = word_embeddings.get_embedding(word)  # Get embedding for each word
                sentence_vectors.append(vector)  # Append the vector to the list

            # Convert the list of numpy arrays to a tensor
            stacked_vectors = torch.stack([torch.FloatTensor(vec) for vec in sentence_vectors])
            
            # Get the average of the word embeddings in the sentence
            sentence_embedding = torch.mean(stacked_vectors, dim=0)
            vectors.append(sentence_embedding)

        # Convert list of sentence embeddings to tensor
        self.embeddings = torch.stack(vectors)  # Shape: [num_sentences, embedding_dim]
        self.labels = torch.tensor(self.labels)
        
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]
        
        
class NN1DAN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, word_embeddings):
        super().__init__()
        
        # Initialize embedding layer using the GloVe embeddings
        self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=True)
        
        # Define the hidden layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Define the output layer
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.relu(x)
        x = self.fc2(x)
        return self.log_softmax(x)