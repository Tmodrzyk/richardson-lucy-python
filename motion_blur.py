import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve
import torch
import torch.nn as nn

# tiny error used for nummerical stability
eps = 0.001


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    """[summary]
    L^2 norm of a list
    [description]
    Used for internals
    Arguments:
        lst {list} -- vector
    """
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]

    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle

    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)


class MotionBlur(nn.Module):
    """[summary]
    Class representing a motion blur kernel of a given intensity.

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

            intensity {float} -- Float between 0 and 1.
            Intensity of the motion blur.

            :   0 means linear motion blur and 1 is a highly non linear
                and often convex motion blur path. (default: {0})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: int = 61, intensity: float=0, channels:int=1):
        super().__init__()

        # check if intensity is float (int) between 0 and 1
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        # saving args
        self.SIZE = (size, size)
        self.kernel_size = size
        self.INTENSITY = intensity

        # deriving quantities

        # we super size first and then downscale at the end for better
        # anti-aliasing
        self.SIZEx2 = tuple([2 * i for i in self.SIZE])
        self.x, self.y = self.SIZEx2

        # getting length of kernel diagonal
        self.DIAGONAL = (self.x**2 + self.y**2)**0.5

        # flag to see if kernel has been calculated already
        self.kernel_is_generated = False

        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, self.kernel_size, stride=1, padding=self.kernel_size//2, padding_mode='replicate', bias=False, groups=channels)
        )
    
        self.weights_init()
    
    def forward(self, x):
        """
        Perform a forward pass of the Gaussian blur operation.

        Args:
            x (torch.Tensor): The input tensor to be blurred.

        Returns:
            torch.Tensor: The blurred output tensor.
        """
        
        return self.seq(x)

    def _createPath(self):
        """[summary]
        creates a motion blur path with the given intensity.
        [description]
        Proceede in 5 steps
        1. Get a random number of random step sizes
        2. For each step get a random angle
        3. combine steps and angles into a sequence of increments
        4. create path out of increments
        5. translate path to fit the kernel dimensions

        NOTE: "random" means random but might depend on the given intensity
        """

        # first we find the lengths of the motion blur steps
        def getSteps():
            """[summary]
            Here we calculate the length of the steps taken by
            the motion blur
            [description]
            We want a higher intensity lead to a longer total motion
            blur path and more different steps along the way.

            Hence we sample

            MAX_PATH_LEN =[U(0,1) + U(0, intensity^2)] * diagonal * 0.75

            and each step: beta(1, 30) * (1 - self.INTENSITY + eps) * diagonal)
            """

            # getting max length of blur motion
            self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * \
                (uniform() + uniform(0, self.INTENSITY**2))

            # getting step
            steps = []

            while sum(steps) < self.MAX_PATH_LEN:

                # sample next step
                step = beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)

            # note the steps and the total number of steps
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            """[summary]
            Gets an angle for each step
            [description]
            The maximal angle should be larger the more
            intense the motion is. So we sample it from a
            U(0, intensity * pi)

            We sample "jitter" from a beta(2,20) which is the probability
            that the next angle has a different sign than the previous one.
            """

            # same as with the steps

            # first we get the max angle in radians
            self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)

            # now we sample "jitter" which is the probability that the
            # next angle has a different sign than the previous one
            self.JITTER = beta(2, 20)

            # initialising angles (and sign of angle)
            angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]

            while len(angles) < self.NUM_STEPS:

                # sample next angle (absolute value)
                angle = triangular(0, self.INTENSITY *
                                   self.MAX_ANGLE, self.MAX_ANGLE + eps)

                # with jitter probability change sign wrt previous angle
                if uniform() < self.JITTER:
                    angle *= - np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])

                angles.append(angle)

            # save angles
            self.ANGLES = np.asarray(angles)

        # Get steps and angles
        getSteps()
        getAngles()

        # Turn them into a path
        ####

        # we turn angles and steps into complex numbers
        complex_increments = polar2z(self.STEPS, self.ANGLES)

        # generate path as the cumsum of these increments
        self.path_complex = np.cumsum(complex_increments)

        # find center of mass of path
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS

        # Shift path s.t. center of mass lies in the middle of
        # the kernel and a apply a random rotation
        ###

        # center it on COM
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex

        # randomly rotate path by an angle a in (0, pi)
        self.path_complex *= np.exp(1j * uniform(0, pi))

        # center COM on center of kernel
        self.path_complex += center_of_kernel

        # convert complex path to final list of coordinate tuples
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def weights_init(self, save_to: Path=None, show: bool=False):
        # check if we haven't already generated a kernel
        if self.kernel_is_generated:
            return None

        # get the path
        self._createPath()

        # Initialise an image with super-sized dimensions
        # (pillow Image object)
        kernel_image = Image.new("RGB", self.SIZEx2)

        # ImageDraw instance that is linked to the kernel image that
        # we can use to draw on our kernel_image
        self.painter = ImageDraw.Draw(kernel_image)

        # draw the path
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism
        self.kernel_image = kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))

        # Resize to actual size
        kernel_image = kernel_image.resize(
            self.SIZE, resample=Image.LANCZOS)

        # convert to gray scale
        kernel_image = kernel_image.convert("L")

        self.k = np.asarray(kernel_image, dtype=np.float32)
        self.k /= np.sum(self.k)
        self.k = torch.from_numpy(self.k)
        
        for name, f in self.named_parameters():
            f.data.copy_(self.k)

    def get_kernel(self):
        """
        Get the Gaussian kernel used for blurring.

        Returns:
            torch.Tensor: The Gaussian kernel.
        """
        return self.k