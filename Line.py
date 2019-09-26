import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Point import Point
from Segment import Segment
from Vector import Vector


class Line:
    def __init__(self, point, vector):
        if len(point) == len(vector):
            self._point = Point(point)
            self._vector = Vector(vector).normalize()
        else:
            raise WrongDimensionException(
                f"Can't init {self.__class__.__name__} using point and vector of different dimensions"
            )

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, point):
        self._point = Point(point)

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = Vector(vector)

    @classmethod
    def from_points(cls, point1, point2):
        return cls(point1, point2 - point1)

    @classmethod
    def from_segment(cls, segment):
        return cls.from_points(segment[0], segment[-1])

    def points(self):
        return [self._point, self._point + self._vector]

    def dim(self):
        return len(self._point)

    def coefficients(self):
        if self.dim() == 2:
            normal = Vector([-self._vector[-1], self._vector[0]])
            return np.array([*normal, -normal.dot(self._point)])
        else:
            raise WrongDimensionException(f"Can't get coefficients of the line in {self.dim()} dimension")

    def __call__(self, point):
        if self.dim() == 2:
            return np.dot(np.append(point, 1.), self.coefficients())
        else:
            raise WrongDimensionException(f"Can't evaluate Line's equation value in {self.dim()} dimension")

    def parameter(self, t):
        return self._point + t*self._vector

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self._point) + ", " + str(self._vector) + ")"

    __repr__ = __str__

    def plot(self, ax, **kwargs):
        if self.dim() == 2:
            coefficients = self.coefficients()
            if coefficients[0] == 0.:
                y = -coefficients[-1] / coefficients[1]
                ax.plot(ax.get_xlim(), [y, y], **kwargs)
            elif coefficients[1] == 0.:
                x = -coefficients[-1] / coefficients[0]
                ax.plot([x, x], ax.get_ylim(), **kwargs)
            else:
                y = -(coefficients[0]*np.array(ax.get_xlim()) + coefficients[-1]) / coefficients[1]
                ax.plot(ax.get_xlim(), y, **kwargs)
        else:
            pass    # TODO: implement Line.plot for 3D

    # TODO: implement Line.from_planes
