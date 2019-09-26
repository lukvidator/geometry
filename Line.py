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

    def param(self, t):
        return self._point + t*self._vector
