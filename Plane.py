import numpy as np
from Exceptions import WrongDimensionException, WrongTypeException
from Point import Point
from tools import extract_coefs, find_three_plane_points
from Vector import Vector


class Plane:
    def __init__(self, point, *vectors):
        if len(point) == len(vectors[0]) and np.array(vectors).shape == 2:
            self._point = np.array(point, dtype=np.float64)
            self._vectors = np.array(vectors, dtype=np.float64)
        else:
            raise WrongDimensionException(f"Can't init {self.__class__} with args of different dimensions")

    @classmethod
    def from_coefficients(cls, coefficients):
        points = find_three_plane_points(coefficients)
        return cls(points[0], [point - points[0] for point in points[1:]])

    @classmethod
    def from_equation(cls, equation, variables):
        return cls.from_coefficients(extract_coefs(equation, variables))

    @classmethod
    def from_points(cls, *points):
        return cls(points[0], [Vector(points[0], point) for point in points[1:]])

    @classmethod
    def from_point_and_normal(cls, point, normal):
        if isinstance(point, Point) and isinstance(normal, Vector):
            return cls(np.append(normal, -np.dot(point, normal)))
        else:
            raise WrongTypeException(f"Can't create {cls.__name__} with {(point, normal)}")

    @property
    def coefficients(self):
        matrix = np.array([self._point, self._vectors])    # creating main matrix
        dim = len(self._point)    # getting the space dimension

        coefficients = np.array(
            [(-1 if i % 2 else 1) * np.linalg.det(
                matrix[np.ix_(range(1, dim), [j for j in range(0, dim) if j != i])]
            ) for i in range(0, dim)])    # creating equation coefficients

        np.append(coefficients, -np.linalg.det(matrix))    # adding free one

        return coefficients

    def equation(self, var=None):
        coefficients = self.coefficients
        if not var:
            equation = \
                ''.join([str(coef) + "*x" + str(i + 1) if coef != 0. else "" for i, coef in enumerate(coefficients[:-1])])
            equation += str(coefficients[-1]) if coefficients[-1] != 0. else ""
            equation.replace("+-", "-")
        elif len(var) == len(coefficients) - 1:
            var = zip(coefficients, var)
            equation = \
                ''.join([str(var[0]) + "*" + var[1] if var[0] != 0. else "" for pair in var])
            equation += str(coefficients[-1]) if coefficients[-1] != 0. else ""
            equation.replace("+-", "-")
        else:
            raise WrongTypeException(f"Can't create equation using {var}")

        return equation

    def normal(self):
        return Vector(self.coefficients[:-1])

    def dim(self):
        return len(self._point)

    @property
    def vectors(self):
        return self._vectors

    @property
    def point(self):
        return self._point
