import tensorflow as tf
import tensorflow.contrib.eager as tfe
import random


class SimpleNN(tf.keras.Model):
    def __init__(self, num_actions, gamma=0.97):
        super(SimpleNN, self).__init__()
        """ Define here the layers used during the forward-pass 
            of the neural network.
        """

        self.gamma = gamma

        # Hidden layer.
        self.dense_layer1 = tf.layers.Dense(30, activation=tf.nn.relu)
        self.dense_layer2 = tf.layers.Dense(20, activation=tf.nn.relu)
        self.dense_layer3 = tf.layers.Dense(10, activation=tf.nn.relu)
        self.output_layer = tf.layers.Dense(num_actions)

    def predict(self, input_data):
        """ Runs a forward-pass through the network.
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).
            Returns:

        """
        hidden_activations = self.dense_layer1(input_data)
        hidden_activations = self.dense_layer2(hidden_activations)
        hidden_activations = self.dense_layer3(hidden_activations)
        logits = self.output_layer(hidden_activations)
        return logits

    def loss_fn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """

        :param input_data: [10x1x5]
        :param target: 10x1
        :return:
        """
        outputs = self.predict(batch_state)  # batch_size x 1 x 3
        batch_action = tf.expand_dims(batch_action, 1)
        batch_action = tf.expand_dims(batch_action, 2)
        outputs = tf.batch_gather(outputs, batch_action)
        outputs = tf.squeeze(outputs, 2)

        next_outputs = self.predict(batch_next_state)
        next_outputs = tf.reduce_max(next_outputs, axis=2)

        target = self.gamma * next_outputs + tf.expand_dims(batch_reward, 1)
        loss = tf.losses.mean_squared_error(labels=outputs, predictions=target)
        # print('Loss  %f' % loss)
        return loss

    def grads_fn(self, batch_state, batch_next_state, batch_reward, batch_action):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(batch_state, batch_next_state, batch_reward, batch_action)
        return tape.gradient(loss, self.variables)


    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        optimizer = tf.train.AdamOptimizer()
        grads = self.grads_fn(batch_state, batch_next_state, batch_reward, batch_action)
        optimizer.apply_gradients(zip(grads, self.variables))
        # print('Loss at epoch %d: %f' % (i + 1, self.loss_fn(input_data, target).numpy()))




#-----------------------------------------------------------------------------------------------------------------------

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
        """
            # initializing lists
            name = [ ("Manjeet",1,1), ("Nikhil",2,2), ("Shambhavi",3,3), ("Astha",4,4) ]
            # using zip() to map values
            mapped = zip(*name)
            # converting values to print as set
            mapped = list(mapped)

        :param batch_size:
        :return:
        """
        samples = random.sample(self.memory, batch_size)
        samples = list(zip(*samples))
        samples =  map(lambda x: tf.convert_to_tensor(x), samples)
        return samples

#-----------------------------------------------------------------------------------------------------------------------

class DeepQLearningNetwork():
    def __init__(self, input_size, num_actions, gamma):


        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        run_config.allow_soft_placement = True
        run_config.log_device_placement = False

        tf.enable_eager_execution(config=run_config)


        self.gamma = gamma
        self.reward_window = []
        self.num_actions = num_actions
        self.model = SimpleNN(num_actions=num_actions, gamma=self.gamma)
        self.memory = ReplayMemory(100000)
        self.last_state = tf.zeros(shape=(1,input_size))
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        actions =  self.model.predict(state)
        actions = tf.nn.softmax(actions * 100)
        action = tf.multinomial(actions, 1)
        return int(action)

    def update(self, reward, new_signal):
        new_state = tf.expand_dims(new_signal, axis=0, name="new_state")

        self.memory.push((self.last_state,
                          new_state,
                          int(self.last_action),
                          self.last_reward))
        action = self.select_action(new_state)

        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.model.learn(batch_state, batch_next_state, batch_reward, batch_action)

        self.last_action = action
        self.last_state = new_state
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