import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.state_size = state_dim
        self.num_actions = action_dim

        self.parameters = []
        self.learning_rate = -1
        self.numTrainingGames = 10000
        self.batch_size = 50

        self.W1 = nn.Parameter(state_dim, 100)
        self.b1 = nn.Parameter(1, 100)
        self.W2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)
        self.W3 = nn.Parameter(100, action_dim)
        self.b3 = nn.Parameter(1, action_dim)

        self.parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        predictedYValues = self.run(states)
        return nn.SquareLoss(predictedYValues, Q_target)


    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"

        linearOutput_1 = nn.Linear(states, self.W1)
        biasedLinear_1 = nn.AddBias(linearOutput_1, self.b1)
        activated_1 = nn.ReLU(biasedLinear_1)

        linearOutput_2 = nn.Linear(activated_1, self.W2)
        biasedLinear_2 = nn.AddBias(linearOutput_2, self.b2)
        activated_2 = nn.ReLU(biasedLinear_2)

        linearOutput_3 = nn.Linear(activated_2, self.W3)
        biasedLinear_3 = nn.AddBias(linearOutput_3, self.b3)

        return biasedLinear_3


    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"

        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)

        for i in range(len(self.parameters)):
            self.parameters[i].update(gradients[i], self.learning_rate)

