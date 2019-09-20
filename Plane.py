import numpy as np
from Exceptions import WrongDimensionException, WrongTypeException
from Point import Point
from tools import extract_coefs, find_three_plane_points
from Vector import Vector


class Plane:
    def __init__(self, *args):
        types = [type(arg) for arg in args]

        if len(args) == 1:
            if type(args[0]) in (tuple, list, np.ndarray):
                self.coefficients = np.array(args[0], dtype=np.float64)
            else:
                raise WrongTypeException(f"Can't create {type(self).__name__} with {args}")
        elif len(args) == 2:
            if types[0] == str and types[1] in (str, tuple, list):
                self._coefs = extract_coefs(*args)
            elif types == [Point, Vector] or types == [np.ndarray, np.ndarray]:
                self._coefs = np.insert(args[1], len(args[1]), -np.dot(*args))
            else:
                raise WrongTypeException(f"Can't create {type(self).__name__} with {args}")
        elif len(args) == 3:
            if types == [Point, Vector, Vector]:
                self._coefs = None    # TODO: implement Plane.__init__ for [Point, Vector, Vector]
            elif types == [Point, Point, Point]:
                self._point = args[0]
                self._vectors = [Vector(args[0], args[1]), Vector(args[0], args[2])]
            else:
                raise WrongTypeException(f"Can't create {type(self).__name__} with {args}")
        else:
            raise WrongTypeException(
                f"{type(self).__name__} __init__ requires 1, 2 or 3 positional arguments but {len(args)} where given"
            )

    @property
    def coefficients(self):
        return self._coefs

    @coefficients.setter
    def coefficients(self, coefs):
        assert type(coefs) in (np.ndarray, list, tuple)
        self._coefs = np.array(coefs, dtype=np.float64)

    def equation(self, var=None):
        if not var:
            equation = \
                ''.join([str(coef) + "*x" + str(i + 1) if coef != 0. else "" for i, coef in enumerate(self._coefs[:-1])])
            equation += str(self._coefs[-1]) if self._coefs[-1] != 0. else ""
            equation.replace("+-", "-")
        elif len(var) == len(self._coefs) - 1:
            var = zip(self._coefs, var)
            equation = \
                ''.join([str(var[0]) + "*" + var[1] if var[0] != 0. else "" for pair in var])
            equation += str(self._coefs[-1]) if self._coefs[-1] != 0. else ""
            equation.replace("+-", "-")
        else:
            raise WrongTypeException(f"Can't create equation using {var}")

        return equation

    def normal(self):
        return Vector(self._coefs[:-1])

    def vectors(self):
        points = find_three_plane_points(self._coefs)
        return [Vector(points[0], points[1]), Vector(points[0], points[2])]
