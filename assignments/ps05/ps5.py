"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2

# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        # state is x, y cords and velocity. acceleration is not used.
        self.state = np.array([init_x, init_y, 0., 0.])  # state

        self.state = self.state.reshape(4, 1)  # reshape to be a single column, 4 rows.

        # initialize the covariance matrix.
        self.covariance = np.matrix(np.diag([1.0, 1.0, 1.0, 1.0]))
        self.Q = Q  # the process noise covariance matrix.
        self.R = R  # the measurement noise covariance matrix.

        # define the transition n x n matrix.
        self.A = np.matrix([
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])
        self.H = np.matrix(np.eye(2, 4)) # measurement matrix.

    def predict(self):
        self.state = np.dot(self.A, self.state)
        self.covariance = np.dot(self.A, np.dot(self.covariance, self.A.T)) + self.Q

    def correct(self, meas_x, meas_y):

        measurement = np.array([meas_x, meas_y]).reshape(2, 1)
        diff = measurement - np.dot(self.H, self.state)
        pred = np.dot(self.H, self.covariance)
        predictive_mean = np.dot(pred, self.H.T) + self.R  # add the noise back in.

        kg = np.dot(self.covariance, np.dot(self.H.T, np.linalg.inv(predictive_mean)))
        self.state = self.state + kg * diff

        # create an identity matrix to set the covariance.
        id = np.eye(self.state.shape[0])
        self.covariance = np.dot(id - np.dot(kg, self.H), self.covariance)

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_mse (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder

        self.template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.frame = frame
        self.particles = None  # We have a better method for initializing our self.particles
        self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles

        # template midpoints for access later.
        self.th, self.tw = self.template.shape[0], self.template.shape[1]

        # the mid point of the template when found. Initialize to just be the middle of the template.
        # given values of x, y, w, h, we know midpoint is x + 1/2 of w and y + 1/2 of h. These values will
        # be updated as teh template moves. These are the estimates for the "template" at each state. We assume that
        # the current state of the template is just in the middle of it
        # self.state_x = self.template_rect['x'] + np.floor(self.template_rect['w'] / 2.)
        # self.state_y = self.template_rect['y'] + np.floor(self.template_rect['h'] / 2.)
        self.state_x = self.template_rect['x'] + self.template_rect['w'] / 2.
        self.state_y = self.template_rect['y'] + self.template_rect['h'] / 2.

        self.state_x = self.state_x if self.state_x % 2 == 0 else self.state_x + 1
        self.state_y = self.state_y if self.state_y % 2 == 0 else self.state_y + 1

        self.th, self.tw = self.template.shape[0], self.template.shape[1]

        # use to keep track of mean errors.
        self.mean_errors = []

    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """

        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_template_around_particle(self, frame, particle, show_image = False):
        """
            Will return a matrix which represents the cutout from teh frame of the template size. The
            matrix is centered around center parameter
        :param frame: The frame to sample the possible template region from
        :param particle: The center aka the particle in which we are analyzing to see if its within the template
        :return: a matrix of template size that contains the pixels from the frame.
        """
        particle = particle.astype(int)
        template = np.zeros((self.th, self.tw), dtype=float)
        h, w = frame.shape[0], frame.shape[1]
        # # determine the stating position in the frame to slice out. The start values is the value of the current
        # # particle - 1/2 the template size. The end value is the current particle + 1/2 the template size.
        fsx, fex = particle[0] - self.tw / 2, particle[0] + self.tw / 2
        fsy, fey = particle[1] - self.th / 2, particle[1] + self.th / 2

        # possible error correction
        fex = fex + (template.shape[1] - (fex - fsx))
        fey = fey + (template.shape[0] - (fey - fsy))
        x_start = 0
        x_end = self.tw
        y_start = 0
        y_end = self.th

        # if the area in which the frame is to be sampled is out of bounds it must be corrected. The new start
        # in the template is offset by the value it is out of bounds. The frame sample is set to either 0 or the
        # max w or h depending on which bound is being checked.
        if fsx < 0:
            x_start = abs(fsx)
            fsx = 0
        elif fex > w:
            fex = w
            x_end = w - fsx

        if fsy < 0:
            y_start = abs(fsy)
            fsy = 0
        elif fey > h:
            fey = h
            y_end = h - fsy

        # print 'template Y range is ({}, {})'.format(y_start, y_end)
        # print 'template x range is ({}, {})'.format(x_start, x_end)
        #
        # print 'frame y range is ({}, {})'.format(fsy, fey)
        # print 'frame x range is ({}, {})'.format(fsx, fex)
        if fsx > w or fex > w or fsy > h or fey > h:
            return template

        if show_image:
            junk_1 = frame[fsy:fey, fsx:fex]
            cv2.imshow('Junk', junk_1)
            cv2.waitKey(0)

        template[y_start:y_end, x_start:x_end] = frame[fsy:fey, fsx:fex]
        return template

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """

        float_template = template.astype(float)
        frame_cutout = frame_cutout.astype(float)
        y, x = float_template.shape[0], float_template.shape[1]

        # for each pixel in temp vs frame, we calulate diff. then square, then sum over each point
        mse = np.sum((float_template - frame_cutout) ** 2) / (y * x)
        return mse

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.

        Returns:
            numpy.array: particles data structure.
        """
        return np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights)

    def distribute_particles(self):
        """
        Sets the particles to a normal random distribution around the current state of the template.
        """
        # np.random.seed(31)
        rx = np.random.normal(loc=self.state_x, scale=self.sigma_dyn, size=self.num_particles) \
            .reshape(self.num_particles, 1)
        ry = np.random.normal(loc=self.state_y, scale=self.sigma_dyn, size=self.num_particles) \
            .reshape(self.num_particles, 1)
        self.particles = np.concatenate((rx, ry), axis=1)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
         """

        self.distribute_particles()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for idx, p in enumerate(self.particles):
            # in the frame, get the area around this particle that can match the template.
            match_area = self.get_template_around_particle(frame, p)
            # get the MSE between the template and the area in the frame
            mse = self.get_error_metric(self.template, match_area)
            self.mean_errors.append(mse)
            # adjust the weights of that particle.
            self.weights[idx] = np.exp(-mse / (2. * self.sigma_exp ** 2.))

        self.weights = self.weights / sum(self.weights)
        self.particles = self.particles[self.resample_particles()]  # updates re-sampled particles

        average_prediction = np.average(self.particles, axis=0, weights=self.weights).astype(np.int)
        self.state_x = average_prediction[0]
        self.state_y = average_prediction[1]

    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        # Complete the rest of the code as instructed.
        x_weighted_mean = x_weighted_mean.astype(int)
        y_weighted_mean = y_weighted_mean.astype(int)

        # dots
        for p in self.particles:
            cv2.circle(frame_in, tuple(p.astype(int)), radius=1, color=(0, 0, 255), thickness=-1)

        # rectangle
        h, w = self.template.shape[:2]
        cv2.rectangle(frame_in,
                      (x_weighted_mean - self.tw / 2, y_weighted_mean - self.th / 2),
                      (x_weighted_mean + w / 2, y_weighted_mean + h / 2),
                      color=(128, 128, 128),
                      thickness=1)

        # circle
        dist = np.linalg.norm(self.particles - [x_weighted_mean, y_weighted_mean])
        radius = np.sum(dist * self.weights.reshape((-1, 1))).astype(int)
        cv2.circle(frame_in, (x_weighted_mean, y_weighted_mean), radius=radius, color=(128, 128, 128), thickness=1)



class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder

    def update_template(self, template):
        self.template = template.astype(np.uint8)
        self.th, self.tw = self.template.shape[0], self.template.shape[1]
        self.distribute_particles()

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)

        # print 'Done processing in regular pf '
        # print 'the current state of the template is at ({}, {})'.format(self.state_x, self.state_y)
        gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        particle = np.array([self.state_x, self.state_y])
        best_frame = self.get_template_around_particle(gray_fr, particle)
        temp_t = self.alpha * best_frame + (1 - self.alpha) * self.template
        self.update_template(temp_t)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.
        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)
        self.scale = 0.9
        self.frames_processed = 0
        self.scale_divisor = 4

        # keep track of last frame to current frame
        self.position_movement = (0, 0)
        # last frame average MSE
        self.avg_mse = 0
        # change from last MSE to current
        self.mse_change = 0
        # max MSE allowed
        self.max_mse = 4200


    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.
        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].
        Returns:
            None.
        """

        # before processing, save off the previous states.
        o_x = self.state_x
        o_y = self.state_y

        # super(MDParticleFilter, self).process(frame)
        # frame = image_resize(frame, width=None, height=int(frame.shape[0] * self.scale))
        ParticleFilter.process(self, frame)

        self.frames_processed += 1
        # determine if there is Occlusion
        current_avg = np.average(self.mean_errors)
        print 'Current avg mse {}'.format(current_avg)
        if current_avg > self.max_mse:
            print 'high mse of {}'.format(current_avg)
            # there is too much error, we assume we want to keep last frames position.
            self.state_x = o_x
            self.state_y = o_y
        elif self.frames_processed % self.scale_divisor == 0 and self.th > 100:
            # it was okay, resample

            # print 'Done processing in regular pf '
            # print 'the current state of the template is at ({}, {})'.format(self.state_x, self.state_y)
            gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            particle = np.array([self.state_x, self.state_y])
            best_frame = self.get_template_around_particle(gray_fr, particle)
            temp_t = self.alpha * best_frame + (1 - self.alpha) * self.template

            resized_temp = image_resize(temp_t, width=None, height=int(self.th * self.scale))
            # resized_temp = cv2.resize(self.template, None, fx=self.scale, fy=self.scale).astype(np.uint8)
            self.scale = self.scale * 0.9
            self.update_template(resized_temp)