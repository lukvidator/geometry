import numpy as np
from Exceptions import WrongDimensionException, WrongTypeException
from Point import Point
from tools import extract_coefs, find_three_plane_points
from Vector import Vector


class Plane:
    def __init__(self, point, vectors):
        if len(point) == len(vectors[0]) and len(np.array(vectors).shape) == 2:
            self._point = Point(point)
            self._vectors = [Vector(vector).normalize() for vector in vectors]
        else:
            raise WrongDimensionException(f"Can't init {self.__class__} with args of different dimensions")

    @classmethod
    def from_coefficients(cls, coefficients):
        return cls.from_points(find_three_plane_points(coefficients))

    @classmethod
    def from_equation(cls, equation, variables):
        return cls.from_coefficients(extract_coefs(equation, variables))

    @classmethod
    def from_points(cls, points):
        return cls(points[0], [Vector(points[0], point) for point in points[1:]])

    @classmethod
    def from_point_and_normal(cls, point, normal):
        if isinstance(point, Point) and isinstance(normal, Vector):
            return cls.from_coefficients(np.append(normal, -np.dot(point, normal)))
        else:
            raise WrongTypeException(f"Can't create {cls.__name__} with {(point, normal)}")

    def coefficients(self):
        matrix = np.array([self._point, *self._vectors])    # creating main matrix
        dim = self.dim()    # getting the space dimension

        coefficients = np.array(
            [(-1 if i % 2 else 1) * np.linalg.det(
                matrix[np.ix_(range(1, dim), [j for j in range(0, dim) if j != i])]
            ) for i in range(0, dim)])    # creating equation coefficients

        coefficients = np.append(coefficients, -np.linalg.det(matrix))    # adding free one

        return coefficients

    def equation(self, var=None):
        coefficients = self.coefficients()
        if not var:
            var = ["x" + str(i + 1) for i in range(0, len(coefficients) - 1)]
        elif not len(var) == len(coefficients) - 1:
            raise WrongTypeException(f"Can't create equation using {var}")

        equation = \
            ' + '.join([str(pair[0]) + "*" + pair[1] if pair[0] != 0. else "" for pair in zip(coefficients, var)])
        equation += " + " + str(coefficients[-1]) if coefficients[-1] != 0. else ""
        equation.replace("+ -", "- ")
        equation += " == 0"

        return equation

    def normal(self):
        return Vector(self.coefficients()[:-1])

    def dim(self):
        return len(self._point)

    @property
    def vectors(self):
        return self._vectors

    @vectors.setter
    def vectors(self, vectors):
        if np.array(vectors).shape == 2:
            self._vectors = [Vector(vector) for vector in vectors]
        else:
            raise WrongDimensionException("Can't set vectors with different dimensions")

    @property
    def point(self):
        return self._point

    @point.setter
    def point(self, point):
        self._point = Point(point)

    def __str__(self):
        if self.dim() < 4:
            return self.__class__.__name__ + "(" + self.equation() + ")"
        else:
            return self.__class__.__name__ + "(" + str(self.point) + ", " + \
                   ", ".join([str(vector) for vector in self._vectors]) + ")"

    __repr__ = __str__

    @staticmethod
    def angle(plane1, plane2):
        normal1 = plane1.normal()
        normal2 = plane2.normal()
        angle = Vector.angle(normal1, normal2)
        return angle if angle <= np.pi/2 else Vector.angle(-normal1, normal2)

