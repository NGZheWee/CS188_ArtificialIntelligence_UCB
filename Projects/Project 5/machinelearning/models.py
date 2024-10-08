import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = self.run(x)
        scoreInFloat = nn.as_scalar(score)
        if scoreInFloat < 0:
            return -1
        else:
            return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        classified = False

        while not classified:
            classified = True  # Assume perfect classification; prove otherwise below

            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    nn.Parameter.update(self.w, x, nn.as_scalar(y))
                    classified = False

            if classified:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.W1 = nn.Parameter(1,100)
        self.b1 = nn.Parameter(1,100)
        self.W2 = nn.Parameter(100,100)
        self.b2 = nn.Parameter(1,100)
        self.W3 = nn.Parameter(100,1)
        self.b3 = nn.Parameter(1,1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # First hidden layer
        linearOutput_1 = nn.Linear(x, self.W1)
        biasedLinear_1 = nn.AddBias(linearOutput_1, self.b1)
        activated_1 = nn.ReLU(biasedLinear_1)

        # Second hidden layer
        linearOutput_2 = nn.Linear(activated_1, self.W2)
        biasedLinear_2 = nn.AddBias(linearOutput_2, self.b2)
        activated_2 = nn.ReLU(biasedLinear_2)

        # Output layer
        linearOutput_3 = nn.Linear(activated_2, self.W3)
        biasedLinear_3 = nn.AddBias(linearOutput_3, self.b3)

        return biasedLinear_3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictedYValues = self.run(x)
        return nn.SquareLoss(predictedYValues, y)  # Use SquareLoss for regression

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        learningRate = -0.01 #negative to minimize loss

        loss = float('inf')
        while loss > 0.02:
            for x, y in dataset.iterate_once(1):
                lossOfNode = self.get_loss(x, y)
                gradients = nn.gradients(lossOfNode, [self.W1, self.b1, self.W2, self.b2,self.W3, self.b3])
                self.W1.update(gradients[0], learningRate)
                self.b1.update(gradients[1], learningRate)
                self.W2.update(gradients[2], learningRate)
                self.b2.update(gradients[3], learningRate)
                self.W3.update(gradients[4], learningRate)
                self.b3.update(gradients[5], learningRate)
            loss = nn.as_scalar(lossOfNode)  #Stop training when the last batch of data meets loss<0.02, might use avg.
            #print("Current loss:", loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.W1 = nn.Parameter(784, 128)
        self.b1 = nn.Parameter(1, 128)
        self.W2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.W3 = nn.Parameter(64, 10)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # First Hidden Layer
        activated_1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))

        # Second hidden layer
        activated_2 = nn.ReLU(nn.AddBias(nn.Linear(activated_1, self.W2), self.b2))

        # Output layer
        logits = nn.AddBias(nn.Linear(activated_2, self.W3), self.b3)

        return logits

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        predictedYValues = self.run(x)
        return nn.SoftmaxLoss(predictedYValues, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        print("Be ready to get bored!")

        learningRate = -0.01 #negative to minimize loss,
        # 0.001, 0.005 too slow

        batch_size = 100 # divisible by dataset size 60000

        while True:
            totalLoss = 0
            for x, y in dataset.iterate_once(batch_size):
                lossOfNode = self.get_loss(x, y)
                gradients = nn.gradients(lossOfNode, [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3])
                self.W1.update(gradients[0], learningRate)
                self.b1.update(gradients[1], learningRate)
                self.W2.update(gradients[2], learningRate)
                self.b2.update(gradients[3], learningRate)
                self.W3.update(gradients[4], learningRate)
                self.b3.update(gradients[5], learningRate)
                totalLoss += nn.as_scalar(lossOfNode)

            avgLoss = totalLoss / dataset.get_validation_accuracy()
            print("Average Loss:", avgLoss)

            # Check validation accuracy
            validationAccuracy = dataset.get_validation_accuracy()
            print("Validation Accuracy:", validationAccuracy)
            if validationAccuracy >= 0.978:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        hidden_size = 150

        self.W1 = nn.Parameter(self.num_chars, hidden_size)
        self.b1 = nn.Parameter(1, hidden_size)
        self.W_hidden = nn.Parameter(hidden_size, hidden_size)
        self.b_hidden = nn.Parameter(1, hidden_size)
        self.W3 = nn.Parameter(hidden_size, len(self.languages))
        self.b3 = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        h = None
        for i in range(len(xs)):
            if i == 0:
                Z = nn.Linear(xs[i], self.W1)
                h = nn.ReLU(nn.AddBias(Z, self.b1))
            else:
                Z = nn.Add(nn.Linear(xs[i], self.W1),
                           nn.Linear(h, self.W_hidden))
                h = nn.ReLU(nn.AddBias(Z, self.b_hidden))
        logits = nn.AddBias(nn.Linear(h, self.W3), self.b3)
        return logits

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        batchSize = 100
        learningRate = -0.12
        epoch = 0
        validationAccuracy = 0

        while validationAccuracy < 0.85:
            for x, y in dataset.iterate_once(batchSize):
                lossofNode = self.get_loss(x, y)
                gradients = nn.gradients(lossofNode, [self.W1, self.b1, self.W_hidden, self.b_hidden, self.W3, self.b3])

                self.W1.update(gradients[0], learningRate)
                self.b1.update(gradients[1], learningRate)
                self.W_hidden.update(gradients[2], learningRate)
                self.b_hidden.update(gradients[3], learningRate)
                self.W3.update(gradients[4], learningRate)
                self.b3.update(gradients[5], learningRate)

            validationAccuracy = dataset.get_validation_accuracy()
            epoch += 1
            print(f'Epoch {epoch}: accuracy = {validationAccuracy}')


