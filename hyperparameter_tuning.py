"""
Find the best classifier using hyperparameter tuning

"""
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import BatchNormalization
import keras.engine.sequential as kes
from keras.wrappers.scikit_learn import KerasClassifier


def create_model(learning_rate=0.01, activation='relu'):
    # Create an Adam optimizer with the given learning rate
    opt = Adam(lr=learning_rate)

    # Create your binary classification model
    model1 = Sequential()
    model1.add(Dense(128, input_shape=(30,), activation=activation))
    model1.add(Dense(256, activation=activation))
    model1.add(Dense(1, activation='sigmoid'))

    # Compile your model with your optimizer, loss, and metrics
    model1.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model1


def get_model(act='relu'):
    return_model = Sequential()

    # Add a hidden layer of 64 neurons and a 20 neuron's input
    return_model.add(Dense(64, activation=act, input_shape=(20,)))

    # Add an output layer of 3 neurons with sigmoid activation
    return_model.add(Dense(3, activation='softmax'))

    # Compile your model with adam and binary crossentropy loss
    return_model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    return return_model


# Replace FILE.csv with the file path.
data_frame = pd.read_csv('FILE.csv')

y = data_frame['label']
X = data_frame.drop(data_frame['label'], axis=1)

# Get data using sklearn train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Get a fresh new model with get_model
standard_model = Sequential()
standard_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
standard_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
standard_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Train your model for 5 epochs with a batch size of 1
standard_model.fit(X_train, y_train, epochs=5, batch_size=1)
print("The accuracy when using a batch of size 1 is: ",
      standard_model.evaluate(X_test, y_test)[1])

# Build your deep network
batchnorm_model = Sequential()
batchnorm_model.add(Dense(50, input_shape=(64,), activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(50, activation='relu', kernel_initializer='normal'))
batchnorm_model.add(BatchNormalization())
batchnorm_model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# Compile your model with sgd
batchnorm_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Standard Network vs BatchNormalization network

# Train your standard model, storing its history
history1 = standard_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Train the batch normalized model you recently built, store its history
history2 = batchnorm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model)

# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 'batch_size': [32, 128, 256],
          'epochs': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}

# Create a randomize search cv object and fit it on the data to obtain the results
random_search = RandomizedSearchCV(model, param_distributions=params, cv=KFold(3))

# random_search.fit(X,y) takes too long! But would start the search.
random_search.fit(X, y)

# Save best parameters
best_epoch = 50
best_batch = 128

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=best_epoch, batch_size=best_batch, verbose=0)

# Calculate the accuracy score for each fold
kfolds = cross_val_score(model, X, y, cv=3)

# Print the mean accuracy and standard deviation
print('The mean accuracy was:', kfolds.mean())
print('With a standard deviation of:', kfolds.std())
