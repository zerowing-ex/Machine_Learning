# Machine_Learning Repository Info
The individual implementations of various algorithms in ML without any external libraries except NumPy.

You can use my code without any limits.

# Implemented Algorithms
## Logistic

## Bayes Nework
### Naive Bayes

## Neural Network
Algorithm design refers to Keras and TensorFlow. The construction way of the NN only supports Function API.

```Python
nn_input = Input(shape=(x_train.shape[1]))
x = Dense(1024, activation='relu')(nn_input)
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
nn_output = Dense(10, activation='softmax')(x)
```

### Activations
- Activation

Abstract class, which is the base class of all activation classes
- Including Indentity, Softmax, Sigmoid, tanh, ReLU, LeakyReLU
### Initializers
- Initializer

Abstract class, which is the base class of all initializer classes

- Including Zeros, Ones, RandomUniform, RandomNormal

### Layers
#### Base Layers
- Layer

Abstract class, which is the base class of all layers

- Input

The input of the nerual network
- Dense

Namely, fully connected layer

#### Normalization Layers
- BatchNormalization

#### Regularization Layers
- Dropout (Not implemented)

### Losses
- LossFunction

Abstract class, which is the base class of all loss functions

#### Probabilistic Losses
- CategoricalCrossentropy
- SparseCategoricalCrossentropy (Not implemented)

#### Regression Losses
- MeanSquaredError (Not implemented)

### Metrics
- Metric

Abstract class, which is the base class of all metrics

- Accuracy

### Optimizers
- Optimizer

Abstract class, which is the base class of all optimizers

- SGD

Stochastic Gradient Descent, momentum and Nestrov is not supported yet

### Model
- Model

compile -> fit -> score/predict (not implemented)

The parameter 'batch_size' is kind of meaningless of CPU training but just for fun :D.

```Python
model = Model(inputs=nn_input, outputs=nn_output)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='acc')
model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=128)
```

### Utils
- to_one_hot function

Convert numpy.ndarray type to the one-hot form

## Support Vector Machine (Not implemented)