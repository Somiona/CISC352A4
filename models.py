import nn


class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        return nn.DotProduct(x_point, self.w)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"

        if nn.as_scalar(self.run(x_point)) >= 0:
            return 1
        else:
            return -1

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"

        batch = 1
        mistake_check = True

        while mistake_check:
            mistake_check = False
            for x, y in dataset.iterate_once(batch):
                predictor = self.get_prediction(x)
                multiplier = nn.as_scalar(y)
                if predictor != multiplier:
                    mistake_check = True
                    self.w.update(multiplier, x)


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.hidden_size = 300
        self.learn_rate = -0.001
        self.threshold = 0.02
        # the batch size is one because any number is divisible by one.
        self.batch = 1

        self.W1 = nn.Parameter(1, self.hidden_size)
        self.W2 = nn.Parameter(self.hidden_size, 1)

        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # The equation used for this block is f(x) = relu(x * W1 + b1) * W2 + b2
        pre_relu = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        relu = nn.ReLU(pre_relu)
        output = nn.AddBias(nn.Linear(relu, self.W2), self.b2)

        return output

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
        # calculate the approximate y value
        approx_y = self.run(x)
        return nn.SquareLoss(approx_y, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        trained = False
        # initially the data is not well trained.
        # the dataset will be trained until the loss of each data is lesser than 0.02
        while not trained:
            trained = True

            for x, y in dataset.iterate_once(self.batch):
                loss = self.get_loss(x, y)

                if nn.as_scalar(loss) > self.threshold:
                    trained = False
                    # this is for getting gradients for each values
                    W1_gr, W2_gr, b1_gr, b2_gr = nn.gradients(
                        [self.W1, self.W2, self.b1, self.b2], loss
                    )

                    # updating each values
                    self.W1.update(self.learn_rate, W1_gr)
                    self.W2.update(self.learn_rate, W2_gr)

                    self.b1.update(self.learn_rate, b1_gr)
                    self.b2.update(self.learn_rate, b2_gr)

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
        # model settings
        # model design is 784 -> 1568 -> 392 -> 196 -> 10 dimensional
        # with ReLU activation function
        self.input_size = 784
        self.hidden_1_size = 1568
        self.hidden_2_size = 392
        self.hidden_3_size = 196
        self.output_size = 10
        # number of maximum epochs
        self.patience = 30
        self.n_batch = 20

        # for gradient descent
        self.lr = -0.005
        self.W1 = nn.Parameter(self.input_size, self.hidden_1_size)
        self.b1 = nn.Parameter(1, self.hidden_1_size)
        self.W2 = nn.Parameter(self.hidden_1_size, self.hidden_2_size)
        self.b2 = nn.Parameter(1, self.hidden_2_size)
        self.W3 = nn.Parameter(self.hidden_2_size, self.hidden_3_size)
        self.b3 = nn.Parameter(1, self.hidden_3_size)
        self.W4 = nn.Parameter(self.hidden_3_size, self.output_size)
        self.b4 = nn.Parameter(1, self.output_size)

        # for momentum gradient descent
        self.use_momentum = False
        self.ita = 0.8
        self.momentum_lr = 0.006
        self.momentum_W1 = nn.Parameter(self.input_size, self.hidden_1_size)
        self.momentum_b1 = nn.Parameter(1, self.hidden_1_size)
        self.momentum_W2 = nn.Parameter(self.hidden_1_size, self.hidden_2_size)
        self.momentum_b2 = nn.Parameter(1, self.hidden_2_size)
        self.momentum_W3 = nn.Parameter(self.hidden_2_size, self.hidden_3_size)
        self.momentum_b3 = nn.Parameter(1, self.hidden_3_size)
        self.momentum_W4 = nn.Parameter(self.hidden_3_size, self.output_size)
        self.momentum_b4 = nn.Parameter(1, self.output_size)

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
        input_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        hidden_layer1 = nn.ReLU(nn.AddBias(nn.Linear(input_layer, self.W2), self.b2))
        hidden_layer2 = nn.ReLU(nn.AddBias(nn.Linear(hidden_layer1, self.W3), self.b3))
        output_layer = nn.AddBias(nn.Linear(hidden_layer2, self.W4), self.b4)

        return output_layer

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
        # calculate the approximate y value
        approx_y = self.run(x)
        return nn.SoftmaxLoss(approx_y, y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Note from Somiona:
        # My motivation of using MGD is that I found my initial implementation of GD
        # have the issue of oscillating around the minimum.
        # So I decide to implement a adaptive learning rate method
        # This method is covered by CISC 372 this term
        # So I think I am okay to use it here
        # To make it work properly, I use nn.Constant class with some hacks
        # targeting nn.py.
        # By default, this is not enabled. But on my computer, this works better.
        if self.use_momentum:
            print("Notice: Using Momentum Gradient Descent")

        def MGD_single(momentum, parameter, gradient):
            # Implementing Momentum Gradient Descent on one parameter
            # Using the formula
            # v_t=\gamma v_{t-1}+\eta \nabla_\theta J(\theta)
            # vₜ=γ vₜ₋₁+η ∇_θ J(θ)
            v_t = self.ita * momentum.data + self.momentum_lr * gradient.data
            # I have to subtract momentum.data from v_t because the update function
            # Says "self.data += multiplier * direction.data"
            # When multiplier is 1, this expands to "self.data = self.data + direction.data"
            # Which is further stepped to "self.data = self.data + v_t - self.data = v_t"
            # Which is exactly what I want
            momentum.update(1, nn.Constant(v_t - momentum.data))
            # \theta=\theta-v_t
            # θ=θ−vₜ
            parameter.update(-1, nn.Constant(momentum.data))

        def MGD_update(W1_gr, b1_gr, W2_gr, b2_gr, W3_gr, b3_gr, W4_gr, b4_gr):
            MGD_single(self.momentum_W1, self.W1, W1_gr)
            MGD_single(self.momentum_b1, self.b1, b1_gr)
            MGD_single(self.momentum_W2, self.W2, W2_gr)
            MGD_single(self.momentum_b2, self.b2, b2_gr)
            MGD_single(self.momentum_W3, self.W3, W3_gr)
            MGD_single(self.momentum_b3, self.b3, b3_gr)
            MGD_single(self.momentum_W4, self.W4, W4_gr)
            MGD_single(self.momentum_b4, self.b4, b4_gr)

        def GD_update(W1_gr, b1_gr, W2_gr, b2_gr, W3_gr, b3_gr, W4_gr, b4_gr):
            self.W1.update(self.lr, W1_gr)
            self.b1.update(self.lr, b1_gr)
            self.W2.update(self.lr, W2_gr)
            self.b2.update(self.lr, b2_gr)
            self.W3.update(self.lr, W3_gr)
            self.b3.update(self.lr, b3_gr)
            self.W4.update(self.lr, W4_gr)
            self.b4.update(self.lr, b4_gr)

        def run_epoch(run_update):
            # from tqdm import tqdm
            # for x, y in tqdm(dataset.iterate_once(self.n_batch), total=60000/self.n_batch):
            for x, y in dataset.iterate_once(self.n_batch):
                loss = self.get_loss(x, y)

                # Calculating gradients for each parameters
                (W1_gr, b1_gr, W2_gr, b2_gr,
                 W3_gr, b3_gr, W4_gr, b4_gr) = nn.gradients(
                    [self.W1, self.b1,
                     self.W2, self.b2,
                     self.W3, self.b3,
                     self.W4, self.b4], loss
                )

                run_update(W1_gr, b1_gr, W2_gr, b2_gr, W3_gr, b3_gr, W4_gr, b4_gr)

        update_func = MGD_update if self.use_momentum else GD_update
        # Run until validation accuracy is greater than 97.6%
        # epoch = 1
        # while (validation := dataset.get_validation_accuracy()) < 0.976 and epoch <= self.patience:
        #     run_epoch(update_func)
        #     print("Epoch: %d, Validation Accuracy: %f" % (epoch, validation))
        #     epoch += 1

        while dataset.get_validation_accuracy() < 0.976 and self.patience > 0:
            run_epoch(update_func)
            self.patience -= 1
