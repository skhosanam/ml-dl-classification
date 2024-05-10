#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install tensorflow')
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[5]:


get_ipython().system('pip install tensorflow')


# In[1]:


import tensorflow as tf
print(tf.__version__)  # This prints the version of TensorFlow to confirm it's installed.


# In[7]:


pip install tensorflow pandas scikit-learn


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate, Layer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[5]:


# Correct the paths and use raw string notation or escape the backslashes
true_path = r"C:\Users\23711055\OneDrive - MMU\Mike_PhD Programme Documents\PhD Datasets of Interest\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\trueNews.csv"
fake_path = r"C:\Users\23711055\OneDrive - MMU\Mike_PhD Programme Documents\PhD Datasets of Interest\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\fakeNews.csv"


# In[10]:


# Custom Layer for Feedback from LSTM to CNN
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Load and preprocess the data
def load_and_preprocess_data(fake_path, true_path):
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])  # Assuming text column is named 'text'
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    
    return data, combined_df['label'].values

# Create the model
def create_model(vocab_size, embedding_dim):
    inputs = Input(shape=(200,))
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    
    lstm = LSTM(64, return_sequences=False)(x)
    
    feedback = FeedbackLayer()([cnn_flat, lstm])
    
    combined = concatenate([feedback, lstm])
    
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training and evaluation
def train_and_evaluate_model(model, data, labels, epochs, batch_size):
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    for epoch in range(epochs):
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=1, validation_data=(val_data, val_labels))

# Main execution flow
true_path = r"C:\Users\23711055\OneDrive - MMU\Mike_PhD Programme Documents\PhD Datasets of Interest\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\trueNews.csv"
fake_path = r"C:\Users\23711055\OneDrive - MMU\Mike_PhD Programme Documents\PhD Datasets of Interest\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\fakeNews.csv"

data, labels = load_and_preprocess_data(fake_path, true_path)
vocab_size = 10000 + 1  # plus one for padding token
embedding_dim = 100
model = create_model(vocab_size, embedding_dim)
train_and_evaluate_model(model, data, labels, epochs=10, batch_size=32)


# In[11]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Custom Layer for Feedback from LSTM to CNN
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Load and preprocess the data
def load_and_preprocess_data(fake_path, true_path):
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])  # Make sure to use 'Text' if that's the column name
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    
    return data, combined_df['label'].values

# Create the model
def create_model(vocab_size, embedding_dim):
    inputs = Input(shape=(200,))
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    
    lstm = LSTM(64, return_sequences=False)(x)
    
    feedback = FeedbackLayer()([cnn_flat, lstm])
    
    combined = concatenate([feedback, lstm])
    
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training and evaluation
def train_and_evaluate_model(model, data, labels, epochs, batch_size):
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    for epoch in range(epochs):
        history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=1, validation_data=(val_data, val_labels))
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        # Format accuracy to 3 decimal places
        print(f"Epoch {epoch + 1}, Train Accuracy: {train_acc:.3f}, Validation Accuracy: {val_acc:.3f}")

# Path to your data files
true_path = r"C:\Users\23711055\OneDrive - MMU\Mike_PhD Programme Documents\PhD Datasets of Interest\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\trueNews.csv"
fake_path = r"C:\Users\23711055\OneDrive - MMU\Mike_PhD Programme Documents\PhD Datasets of Interest\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\fakeNews.csv"

# Instantiate and train the model
vocab_size = 10000 + 1  # plus one for padding token
embedding_dim = 100
model = create_model(vocab_size, embedding_dim)
data, labels = load_and_preprocess_data(fake_path, true_path)
train_and_evaluate_model(model, data, labels, epochs=10, batch_size=32)


# In[ ]:





# In[ ]:


#Hybrid training


# In[21]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)
    def call(self, inputs):
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

def load_and_preprocess_data(fake_path, true_path):
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

def create_model(vocab_size, embedding_dim):
    inputs = Input(shape=(200,))
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    lstm = LSTM(64, return_sequences=False)(x)
    feedback = FeedbackLayer()([cnn_flat, lstm])
    combined = concatenate([feedback, lstm])
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    batch_size_decay = (initial_batch_size - final_batch_size) / (epochs - 1)
    for epoch in range(epochs):
        current_batch_size = int(initial_batch_size - epoch * batch_size_decay)
        num_batches = int(np.ceil(len(train_data) / current_batch_size))
        for batch_index in range(num_batches):
            start_index = batch_index * current_batch_size
            end_index = min((batch_index + 1) * current_batch_size, len(train_data))
            x_batch = train_data[start_index:end_index]
            y_batch = train_labels[start_index:end_index]
            model.train_on_batch(x_batch, y_batch)
        loss, accuracy = model.evaluate(val_data, val_labels, verbose=0)
        print(f"Epoch {epoch + 1}/{epochs}, Batch Size: {current_batch_size}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

data, labels = load_and_preprocess_data(fake_path, true_path)
vocab_size = 10000 + 1
embedding_dim = 100
model = create_model(vocab_size, embedding_dim)

initial_batch_size = 1024
final_batch_size = 32
train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)


# In[ ]:


#Integration of meta learning


# In[32]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

    
  # Define a custom layer that applies feedback from an LSTM output to a CNN layer's output.
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback
     

# Load data and preprocess for training
def load_and_preprocess_data(fake_path, true_path):
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

# Model creation
def create_model(vocab_size, embedding_dim):
    inputs = Input(shape=(200,))
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    lstm = LSTM(64, return_sequences=False)(x)
    feedback = FeedbackLayer()([cnn_flat, lstm])
    combined = concatenate([feedback, lstm])
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callback for dynamic learning rate adjustment
# Callback for dynamic learning rate adjustment
class DynamicLearningRateScheduler(Callback):
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6, max_lr=1e-2):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = self.model.optimizer.learning_rate  # Corrected attribute
                new_lr = max(self.min_lr, current_lr * self.factor)
                new_lr = min(new_lr, self.max_lr)
                self.model.optimizer.learning_rate = new_lr  # Corrected attribute
                print(f"\nEpoch {epoch+1}: Learning rate reduced to {new_lr}.")
                self.wait = 0



# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    dynamic_lr_scheduler = DynamicLearningRateScheduler()
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=[dynamic_lr_scheduler])

# Data paths and loading

true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
model = create_model(vocab_size, embedding_dim)

#data, labels = load_and_preprocess_data('fake_path.csv', 'true_path.csv')
#vocab_size = 10000 + 1
#embedding_dim = 100
#model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)


# In[ ]:





# In[ ]:


#Hybrid batch pro2


# In[33]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Define a custom layer that applies feedback from an LSTM output to a CNN layer's output.
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Load data and preprocess for training
def load_and_preprocess_data(fake_path, true_path):
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

# Model creation
def create_model(vocab_size, embedding_dim):
    inputs = Input(shape=(200,))
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    lstm = LSTM(64, return_sequences=False)(x)
    feedback = FeedbackLayer()([cnn_flat, lstm])
    combined = concatenate([feedback, lstm])
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callback for dynamic learning rate adjustment
class DynamicLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6, max_lr=1e-2):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = self.model.optimizer.learning_rate
                new_lr = max(self.min_lr, current_lr * self.factor)
                new_lr = min(new_lr, self.max_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"\nEpoch {epoch+1}: Learning rate reduced to {new_lr}.")
                self.wait = 0

# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    steps_per_epoch = len(train_data) // initial_batch_size
    
    for epoch in range(epochs):
        batch_size = max(final_batch_size, initial_batch_size - epoch * ((initial_batch_size - final_batch_size) / epochs))
        data_generator = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).shuffle(len(train_data)).batch(int(batch_size))
        
        model.fit(data_generator, epochs=1, verbose=1, validation_data=(val_data, val_labels), callbacks=[DynamicLearningRateScheduler()])

# Data paths and loading
true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)



# In[ ]:


#Everything customised


# In[6]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

# Define a custom layer that applies feedback from an LSTM output to a CNN layer's output.
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define the weights for feedback connections in the layer.
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        # Apply the feedback mechanism by multiplying LSTM output with feedback weights.
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Define Custom Binary Entropy Loss Function
def custom_binary_entropy_loss(y_true, y_pred):
    # Custom implementation of binary cross-entropy loss
    # You can adjust the loss calculation based on your requirements
    binary_loss = -1 * (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return binary_loss

# Define Custom Optimizer Function
class CustomOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, custom_param=0.001, **kwargs):
        super(CustomOptimizer, self).__init__(**kwargs)
        self.custom_param = custom_param

    def get_updates(self, loss, params):
        # Custom update rule for adjusting optimizer parameters
        updates = super().get_updates(loss, params)
        # Modify updates based on custom parameters
        # You can adjust the optimizer behavior here
        return updates

# Load data and preprocess for training
def load_and_preprocess_data(fake_path, true_path):
    # Load fake and true news data from CSV files.
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    # Assign labels to fake (1) and true (0) news.
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    # Concatenate fake and true news data into a single DataFrame.
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    # Tokenize text data and pad sequences to a fixed length.
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

# Model creation
def create_model(vocab_size, embedding_dim):
    # Define input layer for text data.
    inputs = Input(shape=(200,))
    # Embedding layer for word embeddings.
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    # CNN layer for extracting local features.
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    # LSTM layer for capturing temporal dependencies.
    lstm = LSTM(64, return_sequences=False)(x)
    # Feedback layer to integrate LSTM output with CNN output.
    feedback = FeedbackLayer()([cnn_flat, lstm])
    # Concatenate feedback layer output with LSTM output.
    combined = concatenate([feedback, lstm])
    # Dense layer for classification.
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    # Define the model architecture.
    model = Model(inputs=inputs, outputs=outputs)
    # Compile the model with custom loss function and optimizer
    model.compile(optimizer=CustomOptimizer(custom_param=0.001), loss=custom_binary_entropy_loss, metrics=['accuracy'])
    return model

# Callback for dynamic learning rate adjustment
class DynamicLearningRateScheduler(Callback):
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6, max_lr=1e-2):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Monitor validation loss for early stopping.
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Adjust learning rate if patience threshold is reached.
                current_lr = self.model.optimizer.learning_rate
                new_lr = max(self.min_lr, current_lr * self.factor)
                new_lr = min(new_lr, self.max_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"\nEpoch {epoch+1}: Learning rate reduced to {new_lr}.")
                self.wait = 0

# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    # Split data into training and validation sets.
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Initialize dynamic learning rate scheduler callback.
    dynamic_lr_scheduler = DynamicLearningRateScheduler()
    # Fit the model to the training data, validating on the validation data.
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=[dynamic_lr_scheduler])

# Data paths and loading
true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

# Load and preprocess data from CSV files.
data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
# Create the neural network model.
model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)


# In[ ]:





# In[5]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np

# Define a custom layer that applies feedback from an LSTM output to a CNN layer's output.
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define the weights for feedback connections in the layer.
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        # Apply the feedback mechanism by multiplying LSTM output with feedback weights.
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Load data and preprocess for training
def load_and_preprocess_data(fake_path, true_path):
    # Load fake and true news data from CSV files.
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    # Assign labels to fake (1) and true (0) news.
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    # Concatenate fake and true news data into a single DataFrame.
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    # Tokenize text data and pad sequences to a fixed length.
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

# Define Custom Binary Entropy Loss Function with added epsilon for numerical stability
def custom_binary_entropy_loss(y_true, y_pred):
    epsilon = 1e-7  # Add a small epsilon value to prevent log(0)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0) or log(1)
    binary_loss = -1 * (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return binary_loss

# Define Custom Optimizer with adjusted learning rate
class CustomOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, custom_param=0.001, **kwargs):
        super(CustomOptimizer, self).__init__(**kwargs)
        self.custom_param = custom_param

    def get_updates(self, loss, params):
        updates = super().get_updates(loss, params)
        # Adjust learning rate based on custom parameters
        new_lr = self.lr * self.custom_param  # Adjust learning rate using a custom parameter
        self.lr = K.clip(new_lr, self.min_lr, self.max_lr)  # Clip learning rate to min_lr and max_lr
        return updates

# Model creation
def create_model(vocab_size, embedding_dim):
    # Define input layer for text data.
    inputs = Input(shape=(200,))
    # Embedding layer for word embeddings.
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    # CNN layer for extracting local features.
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    # LSTM layer for capturing temporal dependencies.
    lstm = LSTM(64, return_sequences=False)(x)
    # Feedback layer to integrate LSTM output with CNN output.
    feedback = FeedbackLayer()([cnn_flat, lstm])
    # Concatenate feedback layer output with LSTM output.
    combined = concatenate([feedback, lstm])
    # Dense layer for classification.
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    # Define the model architecture.
    model = Model(inputs=inputs, outputs=outputs)
    # Compile the model with custom optimizer, loss function, and evaluation metrics.
    model.compile(optimizer=CustomOptimizer(custom_param=0.1), loss=custom_binary_entropy_loss, metrics=['accuracy'])
    return model

# Callback for dynamic learning rate adjustment
class DynamicLearningRateScheduler(Callback):
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6, max_lr=1e-2):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Monitor validation loss for early stopping.
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Adjust learning rate if patience threshold is reached.
                current_lr = self.model.optimizer.learning_rate
                new_lr = max(self.min_lr, current_lr * self.factor)
                new_lr = min(new_lr, self.max_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"\nEpoch {epoch+1}: Learning rate reduced to {new_lr}.")
                self.wait = 0

# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    # Split data into training and validation sets.
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Initialize dynamic learning rate scheduler callback.
    dynamic_lr_scheduler = DynamicLearningRateScheduler()
    # Fit the model to the training data, validating on the validation data.
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=[dynamic_lr_scheduler])

# Data paths and loading
true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

# Load and preprocess data from CSV files.
data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
# Create the neural network model with adjusted custom loss function and optimizer
model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)


# In[ ]:





# In[10]:


import matplotlib.pyplot as plt

# Define the training metrics
accuracy = [0.9256, 0.9985, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
loss = [0.1790, 0.0042, 2.9416e-04, 1.5356e-05, 8.8133e-06, 8.8124e-06, 6.2021e-06, 5.0087e-06, 4.5843e-06, 3.9860e-06]
val_accuracy = [0.9934, 0.9934, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960, 0.9960]
val_loss = [0.0217, 0.0216, 0.0186, 0.0190, 0.0194, 0.0195, 0.0197, 0.0198, 0.0198, 0.0199]

# Plotting accuracy and validation accuracy against epochs
epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(12, 6))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'b', label='Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# In[11]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a custom layer that applies feedback from an LSTM output to a CNN layer's output.
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define the weights for feedback connections in the layer.
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        # Apply the feedback mechanism by multiplying LSTM output with feedback weights.
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Load data and preprocess for training
def load_and_preprocess_data(fake_path, true_path):
    # Load fake and true news data from CSV files.
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    # Assign labels to fake (1) and true (0) news.
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    # Concatenate fake and true news data into a single DataFrame.
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    # Tokenize text data and pad sequences to a fixed length.
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

# Define Custom Binary Entropy Loss Function with added epsilon for numerical stability
def custom_binary_entropy_loss(y_true, y_pred):
    epsilon = 1e-7  # Add a small epsilon value to prevent log(0)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0) or log(1)
    binary_loss = -1 * (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return binary_loss

# Define Custom Optimizer with adjusted learning rate
class CustomOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, custom_param=0.001, **kwargs):
        super(CustomOptimizer, self).__init__(**kwargs)
        self.custom_param = custom_param

    def get_updates(self, loss, params):
        updates = super().get_updates(loss, params)
        # Adjust learning rate based on custom parameters
        new_lr = self.lr * self.custom_param  # Adjust learning rate using a custom parameter
        self.lr = K.clip(new_lr, self.min_lr, self.max_lr)  # Clip learning rate to min_lr and max_lr
        return updates

# Model creation
def create_model(vocab_size, embedding_dim):
    # Define input layer for text data.
    inputs = Input(shape=(200,))
    # Embedding layer for word embeddings.
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    # CNN layer for extracting local features.
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    # LSTM layer for capturing temporal dependencies.
    lstm = LSTM(64, return_sequences=False)(x)
    # Feedback layer to integrate LSTM output with CNN output.
    feedback = FeedbackLayer()([cnn_flat, lstm])
    # Concatenate feedback layer output with LSTM output.
    combined = concatenate([feedback, lstm])
    # Dense layer for classification.
    x = Dense(64, activation='relu')(combined)
    outputs = Dense(1, activation='sigmoid')(x)
    # Define the model architecture.
    model = Model(inputs=inputs, outputs=outputs)
    # Compile the model with custom optimizer, loss function, and evaluation metrics.
    model.compile(optimizer=CustomOptimizer(custom_param=0.1), loss=custom_binary_entropy_loss, metrics=['accuracy'])
    return model

# Callback for dynamic learning rate adjustment
class DynamicLearningRateScheduler(Callback):
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6, max_lr=1e-2):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Monitor validation loss for early stopping.
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Adjust learning rate if patience threshold is reached.
                current_lr = self.model.optimizer.learning_rate
                new_lr = max(self.min_lr, current_lr * self.factor)
                new_lr = min(new_lr, self.max_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"\nEpoch {epoch+1}: Learning rate reduced to {new_lr}.")
                self.wait = 0

# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    # Split data into training and validation sets.
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Initialize dynamic learning rate scheduler callback.
    dynamic_lr_scheduler = DynamicLearningRateScheduler()
    # Fit the model to the training data, validating on the validation data.
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=[dynamic_lr_scheduler])
    return history

# Data paths and loading
true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

# Load and preprocess data from CSV files.
data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
# Create the neural network model with adjusted custom loss function and optimizer
model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
history = train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)

# Plotting accuracy and validation accuracy against epochs
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(12, 6))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'b', label='Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[14]:


import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a custom layer that applies feedback from an LSTM output to a CNN layer's output.
class FeedbackLayer(Layer):
    def __init__(self, **kwargs):
        super(FeedbackLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define the weights for feedback connections in the layer.
        self.feedback_weights = self.add_weight(
            shape=(input_shape[1][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        # Apply the feedback mechanism by multiplying LSTM output with feedback weights.
        system1_output, system2_output = inputs
        feedback = tf.matmul(system2_output, self.feedback_weights)
        return system1_output + feedback

# Load data and preprocess for training
def load_and_preprocess_data(fake_path, true_path):
    # Load fake and true news data from CSV files.
    fake_news_df = pd.read_csv(fake_path)
    true_news_df = pd.read_csv(true_path)
    # Assign labels to fake (1) and true (0) news.
    fake_news_df['label'] = 1
    true_news_df['label'] = 0
    # Concatenate fake and true news data into a single DataFrame.
    combined_df = pd.concat([fake_news_df, true_news_df], axis=0)
    # Tokenize text data and pad sequences to a fixed length.
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(combined_df['Text'])
    sequences = tokenizer.texts_to_sequences(combined_df['Text'])
    data = pad_sequences(sequences, maxlen=200)
    return data, combined_df['label'].values

# Define Custom Binary Entropy Loss Function with added epsilon for numerical stability
def custom_binary_entropy_loss(y_true, y_pred):
    epsilon = 1e-7  # Add a small epsilon value to prevent log(0)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0) or log(1)
    binary_loss = -1 * (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return binary_loss

# Define Custom Optimizer with adjusted learning rate
class CustomOptimizer(tf.keras.optimizers.Adam):
    def __init__(self, custom_param=0.001, **kwargs):
        super(CustomOptimizer, self).__init__(**kwargs)
        self.custom_param = custom_param

    def get_updates(self, loss, params):
        updates = super().get_updates(loss, params)
        # Adjust learning rate based on custom parameters
        new_lr = self.lr * self.custom_param  # Adjust learning rate using a custom parameter
        self.lr = K.clip(new_lr, self.min_lr, self.max_lr)  # Clip learning rate to min_lr and max_lr
        return updates

# Model creation
def create_model(vocab_size, embedding_dim):
    # Define input layer for text data.
    inputs = Input(shape=(200,))
    # Embedding layer for word embeddings.
    x = Embedding(vocab_size, embedding_dim, input_length=200)(inputs)
    # CNN layer for extracting local features.
    cnn = Conv1D(32, 5, activation='relu')(x)
    cnn = MaxPooling1D(5)(cnn)
    cnn_flat = Flatten()(cnn)
    # LSTM layer for capturing temporal dependencies.
    lstm = LSTM(64, return_sequences=False)(x)
    # Feedback layer to integrate LSTM output with CNN output.
    feedback = FeedbackLayer()([cnn_flat, lstm])
    # Concatenate feedback layer output with LSTM output.
    combined = concatenate([feedback, lstm])
    # Dense layer for classification.
    x = Dense(64, activation='relu')(combined)
    # Adding a dropout layer to address overfitting
    x = Dropout(0.5)(x)  # Dropout rate of 0.5
    outputs = Dense(1, activation='sigmoid')(x)
    # Define the model architecture.
    model = Model(inputs=inputs, outputs=outputs)
    # Compile the model with custom optimizer, loss function, and evaluation metrics.
    model.compile(optimizer=CustomOptimizer(custom_param=0.1), loss=custom_binary_entropy_loss, metrics=['accuracy'])
    return model

# Callback for dynamic learning rate adjustment
class DynamicLearningRateScheduler(Callback):
    def __init__(self, factor=0.5, patience=2, min_lr=1e-6, max_lr=1e-2):
        super(DynamicLearningRateScheduler, self).__init__()
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.wait = 0
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        # Monitor validation loss for early stopping.
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # Adjust learning rate if patience threshold is reached.
                current_lr = self.model.optimizer.learning_rate
                new_lr = max(self.min_lr, current_lr * self.factor)
                new_lr = min(new_lr, self.max_lr)
                self.model.optimizer.learning_rate = new_lr
                print(f"\nEpoch {epoch+1}: Learning rate reduced to {new_lr}.")
                self.wait = 0

# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    # Split data into training and validation sets.
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Initialize dynamic learning rate scheduler callback.
    dynamic_lr_scheduler = DynamicLearningRateScheduler()
    # Fit the model to the training data, validating on the validation data.
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=[dynamic_lr_scheduler])
    return history

# Data paths and loading
true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

# Load and preprocess data from CSV files.
data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
# Create the neural network model with adjusted custom loss function and optimizer
model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
history = train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)

# Plotting accuracy and validation accuracy against epochs
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(12, 6))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'b', label='Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[15]:


import matplotlib.pyplot as plt

# Training and evaluation with dynamic batch size and learning rate
def train_and_evaluate_model(model, data, labels, epochs, initial_batch_size, final_batch_size):
    # Split data into training and validation sets.
    train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    # Initialize dynamic learning rate scheduler callback.
    dynamic_lr_scheduler = DynamicLearningRateScheduler()
    # Fit the model to the training data, validating on the validation data.
    history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=[dynamic_lr_scheduler])
    return history

# Data paths and loading
true_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\trueNews.csv"
fake_path = "C:\\Users\\23711055\\OneDrive - MMU\\Mike_PhD Programme Documents\\PhD Datasets of Interest\\Covid-19 Fake News Infodemic Research (CoVID19-FNIR)\\fakeNews.csv"

# Load and preprocess data from CSV files.
data, labels = load_and_preprocess_data(fake_path, true_path)

# Model parameters and training
vocab_size = 10000 + 1
embedding_dim = 100
# Create the neural network model with adjusted custom loss function and optimizer
model = create_model(vocab_size, embedding_dim)

# Execute training with dynamic batch sizes and learning rate adjustments
initial_batch_size = 1024
final_batch_size = 32
history = train_and_evaluate_model(
    model=model,
    data=data,
    labels=labels,
    epochs=10,
    initial_batch_size=initial_batch_size,
    final_batch_size=final_batch_size
)

# Plotting accuracy and validation accuracy against epochs
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(12, 6))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'b', label='Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


#These plots allowing for the examination of the learning curve of the model. 
#If the training accuracy continues to increase while the validation accuracy plateaus or decreases, it might indicate overfitting. 
#Similarly, if the training loss continues to decrease while the validation loss starts to increase, it could also indicate overfitting.

