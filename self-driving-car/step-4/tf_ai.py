import tensorflow as tf
import tensorflow.contrib.eager as tfe
import random


class SimpleNN(tf.keras.Model):
    def __init__(self, num_actions):
        super(SimpleNN, self).__init__()
        """ Define here the layers used during the forward-pass 
            of the neural network.
        """
        # Hidden layer.
        self.dense_layer1 = tf.layers.Dense(30, activation=tf.nn.relu)
        self.dense_layer2 = tf.layers.Dense(20, activation=tf.nn.relu)
        self.dense_layer3 = tf.layers.Dense(10, activation=tf.nn.relu)
        self.output_layer = tf.layers.Dense(num_actions, activation=tf.nn.softmax)

    def predict(self, input_data):
        """ Runs a forward-pass through the network.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:

        """
        hidden_activations = self.dense_layer(input_data)
        hidden_activations = self.dense_layer(hidden_activations)
        hidden_activations = self.dense_layer(hidden_activations)
        actions = self.output_layer(hidden_activations)
        return actions

    def loss_fn(self, input_data, target):
        """ Defines the loss function used during
            training.
        """
        logits = self.predict(input_data)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=logits)
        return loss

    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)

    def fit(self, input_data, target, optimizer, num_epochs=1, verbose=50):
        """ Function to train the model, using the selected optimizer and
            for the desired number of epochs.
        """
        for i in range(num_epochs):
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i == 0) | ((i + 1) % verbose == 0):
                print('Loss at epoch %d: %f' % (i + 1, self.loss_fn(input_data, target).numpy()))

# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] # [[last_state, new_state, last_action, last_reward]...]

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        samples = tf.convert_to_tensor(samples, dtype=tf.float32)
        return samples

class DeepQLearningNetwork():
    def __init__(self, input_size, num_actions, gamma):
        tf.enable_eager_execution()
        self.gamma = gamma
        self.reward_window = []
        self.num_actions = num_actions
        self.model = SimpleNN(num_actions=num_actions)
        self.memory = ReplayMemory(100000)
        self.last_state = tf.zeros(shape=(1,input_size))
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        actions =  self.model.predict(state)
        action = tf.argmax(actions)
        return action

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        pass

    def update(self, reward, new_signal):
        self.memory.push((self.last_state,
                          new_signal,
                          self.last_action,
                          self.last_reward))
        action = self.select_action(new_signal)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_signal
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        pass

    def load(self):
        pass