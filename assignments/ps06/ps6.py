"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   ([int]): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    size = tuple(size)
    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    x = []
    y = []
    for i in images_files:
        img = cv2.imread(folder + '/' + i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # docs state that InterCubic is slower, but generally will look better. If performance
        # becomes an issue, swap out.
        img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)  # resize to param size.
        x.append(img.flatten())
        # Label for the image.
        # https://stackoverflow.com/questions/8143363/convert-string-into-integer
        label = i.split('.')[0]
        label = int(filter(str.isdigit, label))
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    return x, y


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """

    M = X.shape[0]  # the count of images.
    training_sample_count = int(p * M)

    idx = np.random.permutation(X.shape[0]) # get a random list of indices.

    # training index will be from 0 to training sample count.
    training_idx = idx[: training_sample_count]
    # testing will be from count to end
    testing_idx = idx[training_sample_count:]

    Xtrain = X[training_idx, :]
    ytrain = y[training_idx]
    Xtest = X[testing_idx, :]
    ytest = y[testing_idx]
    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x, axis=0, dtype=float)


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    mean = get_mean_face(X)
    # subtract the mean from the faces.
    phi = X - mean
    # the value right of the sigma in our assignment pdf.
    u = np.dot(phi.T, phi)
    eigenvalues, eigenvectors = np.linalg.eigh(u)
    # sort
    sorted_e_values = np.argsort(eigenvalues)
    # get the top k vectors.
    top_e_vectors = sorted_e_values[::-1][:k]
    eigenvectors = eigenvectors[:, top_e_vectors]
    eigenvalues = eigenvalues[top_e_vectors]
    return eigenvectors, eigenvalues


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""

        for i in range(self.num_iterations):
            # re normalize the weights so they sum to 1.
            self.weights /= np.sum(self.weights)
            # instantiate the weak classifier h
            h = WeakClassifier(X=self.Xtrain, y=self.ytrain, weights=self.weights)
            h.train()
            # get predictions for all the values in x training.
            predictions = h.predict(np.transpose(self.Xtrain))

            # find ej for weights
            # sum weights where the prediction label does not equal the expected label.
            ej = np.sum([self.weights[i] for i in range(len(predictions)) if predictions[i] != self.ytrain[i]])
            # calculate alpha
            alpha = 0.5 * np.log((1-ej) / ej)
            self.weakClassifiers.append(h)
            self.alphas.append(alpha)
            if ej > self.eps:
                # the weights are now going to be adjusted for error weights.
                self.weights = [self.weights[i] * np.exp(-self.ytrain[i] * alpha * predictions[i]) for i in range(len(self.weights))]
            else:  # stop the loop
                return

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        correct = 0
        incorrect = 0
        predictions = self.predict(self.Xtrain)

        for p in range(len(predictions)):
            if predictions[p] == self.ytrain[p]:
                correct += 1
            else:
                incorrect += 1

        return correct, incorrect

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        predictions = []

        # for each weak classifier, get the list of predictions and store that in an overall array. We will
        # need to have this so alpha can be applied after the predictions are made.
        for i in range(len(self.weakClassifiers)):
            p = [self.weakClassifiers[i].predict(np.transpose(X))]
            predictions.append(p)

        # apply alphas to the predictions that were made.
        for i in range(len(self.alphas)):
            predictions[i] = np.array(predictions[i]) * self.alphas[i]

        # sum the predictions across all the weak classifiers.
        predictions = np.sum(predictions, axis=0)[0]
        return np.sign(predictions)


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        img = np.zeros(shape, dtype=np.uint8)
        # area of addition is white (255)
        img[y: y + int(h/2), x: x + w] = 255
        # area of subtraction is gray (126)
        img[y + int(h/2): y + h, x: x+ w] = 126
        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        img = np.zeros(shape, dtype=np.uint8)
        # area of addition is white (255)
        img[y: y + h, x: x + int(w/2)] = 255
        # area of subtraction is gray (126)
        img[y: y + h, x + int(w/2): x + w] = 126
        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        img = np.zeros(shape, dtype=np.uint8)
        # area of addition is white (255)
        img[y: y + int(h / 3), x: x + w] = 255
        # area of subtraction is gray ( 126)
        img[y + int(h/3): y + (2 * int(h/3)), x: x + w] = 126
        # area of addition is white (255)
        img[y + (2 * int(h/3)): y + h, x: x + w] = 255
        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        img = np.zeros(shape, dtype=np.uint8)
        # area of addition is white (255)
        img[y: y + h, x: x + int(w/3)] = 255
        # area of subtraction is gray ( 126)
        img[y: y + h, x + int(w/3): x + (2 * int(w/3))] = 126
        # area of addition is white (255)
        img[y: y + h, x + (2 * int(w/3)): x + w] = 255
        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        y, x = self.position
        h, w = self.size
        img = np.zeros(shape, dtype=np.uint8)
        # top left
        img[y: y + int(h/2), x: x + int(w / 2)] = 126
        # top right
        img[y: y + int(h/2), x + int(w/2): x + w] = 255

        # bottom left
        img[y + int(h/2): y + h, x: x + int(w / 2)] = 255
        # bottom right
        img[y + int(h / 2): y + h, x + int(w/2): x + w] = 126
        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """

        # cast to prevent issues related to the following output when running:
        # RuntimeWarning: overflow encountered in ulong_scalars
        ii = np.float64(ii)
        # init return value to 0
        score = 0

        y, x = self.position
        h, w = self.size

        y = y - 1
        x = x - 1
        # keeping it in the bounds.
        y = y if y >= 0 else 0
        x = x if x >= 0 else 0

        if self.feat_type == (2, 1):
            # horizontal lines

            # top
            a = ii[y, x]
            b = ii[y + (h//2), x]
            c = ii[y, x + w]
            d = ii[y + (h//2), x + w]
            score += d - b - c + a

            # bottom
            a = ii[y + (h//2), x]
            b = ii[y + h, x]
            c = ii[y + (h//2), x + w]
            d = ii[y + h, x + w]
            score -= d - b - c + a

        elif self.feat_type == (1, 2):

            a = ii[y, x]
            b = ii[y + h, x]
            c = ii[y, x + (w // 2)]
            d = ii[y + h, x + (w // 2)]
            score += d - b - c + a

            a = ii[y, x + (w // 2)]
            b = ii[y + h, x + (w // 2)]
            c = ii[y, x + w]
            d = ii[y + h, x + w]
            score -= d - b - c + a

        elif self.feat_type == (3, 1):

            # White Region
            a = ii[y, x]
            b = ii[y + (h//3), x]
            c = ii[y, x + w]
            d = ii[y + (h//3), x + w]
            score += d - b - c + a

            # Gray Region
            a = ii[y + (h//3), x]
            b = ii[y + (2 * (h//3)), x]
            c = ii[y + (h//3), x + w]
            d = ii[y + (2 * (h//3)), x + w]
            score -= d - b - c + a

            # White Region
            a = ii[y + (2 * (h//3)), x]
            b = ii[y + h, x]
            c = ii[y + (2 * (h//3)), x + w]
            d = ii[y + h, x + w]
            score += d - b - c + a

        elif self.feat_type == (1, 3):
            # vertical lines

            # White Region
            a = ii[y, x]
            b = ii[y + h, x]
            c = ii[y, x + (w // 3)]
            d = ii[y + h, x + (w // 3)]
            score += d - b - c + a

            # Gray Region
            a = ii[y, x + (w // 3)]
            b = ii[y + h, x + (w // 3)]
            c = ii[y, x + (2 * (w//3))]
            d = ii[y + h, x + (2 * (w//3))]
            score -= d - b - c + a

            # White Region
            a = ii[y, x + (2 * (w//3))]
            b = ii[y + h, x + (2 * (w//3))]
            c = ii[y, x + w]
            d = ii[y + h, x + w]
            score += d - b - c + a

        else:
            # square
            a = ii[y, x]
            b = ii[y + (h//2), x]
            c = ii[y, x + (w // 2)]
            d = ii[y + (h//2), x + (w // 2)]
            score -= d - b - c + a

            a = ii[y, x + (w // 2)]
            b = ii[y + (h//2), x + (w // 2)]
            c = ii[y, x + w]
            d = ii[y + (h//2), x + w]
            score += d - b - c + a

            a = ii[y + (h//2), x]
            b = ii[y + h, x]
            c = ii[y + (h//2), x + (w // 2)]
            d = ii[y + h, x + (w // 2)]
            score += d - b - c + a

            a = ii[y + (h//2), x + (w // 2)]
            b = ii[y + h, x + (w // 2)]
            c = ii[y + (h//2), x + w]
            d = ii[y + h, x + w]
            score -= d - b - c + a

        return score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """

    ii = []
    # loop over each image that is passed and cumulative sum
    for i in images:
        i = np.cumsum(i, axis=0)
        i = np.cumsum(i, axis=1)
        ii.append(i)
    return ii


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.iteritems():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print " -- compute all scores --"
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print " -- select classifiers --"
        for i in range(num_classifiers):

            # normalize weights.
            weights = weights / np.sum(weights)
            # instantiate each classifier
            classifier = VJ_Classifier(scores, self.labels, weights)
            classifier.train()
            error = classifier.error
            # append
            self.classifiers.append(classifier)
            # update weights.
            B = error / (1.0 - error)
            a = np.log(1.0 / B)

            # get predictions
            preds = classifier.predict(np.transpose(scores))
            # e = -1 if same, 1 if not.
            e = [-1 if preds[i] == self.labels[i] else 1 for i in range(len(preds))]

            for wti in range(len(weights)):
                weights[wti] = weights[wti] * np.power(B, 1-e[wti])

            # append the alpha.
            self.alphas.append(a)


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        for clf in self.classifiers:
            feature_id = clf.feature
            feature = self.haarFeatures[feature_id]
            # for each image in the integral images.
            for x in range(len(ii)):
                scores[x, feature_id] = feature.evaluate(ii[x])

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        for x in scores:

            lhs = np.sum([self.alphas[i] * self.classifiers[i].predict(x) for i in range(len(self.alphas))])
            rhs = 0.5 * np.sum(self.alphas)
            if lhs >= rhs:
                result.append(1)
            else:
                result.append(-1)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        ws = 24
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images = []
        windows = []

        # for each row, starting at 12, step of 12. This is due to
        for y in range(ws/2, image.shape[0], ws/2):
            # for each column starting at 12, step of 12
            for x in range(ws/2, image.shape[1], ws/2):
                img_slice = img[y-(ws/2): y + (ws/2), x-(ws/2): x + (ws/2)]
                windows.append((y, x))
                images.append(img_slice)

        preds = self.predict(images)

        for i in range(len(preds)):
            if preds[i] == 1:
                cv2.rectangle(image, (windows[i][1] - (ws/2), windows[i][0] - (ws/2)),
                              (windows[i][1] + (ws/2), windows[i][0] + (ws/2)),
                              (0, 0, 255), 1)

        cv2.imwrite('output/' + filename + '.png', image)
