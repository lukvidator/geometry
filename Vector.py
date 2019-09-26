from Point import Point
import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException


class Vector(Point):
    def __init__(self, point1: Point, point2=Point([0, 0, 0]), dtype=np.float64):
        self._coord = np.array(point1, dtype=dtype) - np.array(point2, dtype=dtype)

    def __neg__(self):
        return Vector(-self._coord)

    def __add__(self, other):
        return Vector(super(Vector, self).__add__(other))

    def __sub__(self, other):
        try:
            return self.__add__(-other)
        except (WrongTypeException, AttributeError):
            raise WrongTypeException(
                f"Can't subtract an object of type {type(other).__name__} from {type(self).__name__}"
            )
        except WrongDimensionException:
            raise WrongDimensionException(
                f"subtract {type(other).__name__} of len {len(other)} from {type(self).__name__} of len {len(self)}"
            )

    def __mul__(self, other):
        return Vector(self._coord * other)

    def __truediv__(self, other):
        return Vector(self._coord / other)

    __rmul__ = __mul__
    __iadd__ = __add__
    __radd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __itruediv__ = __truediv__

    def dot(self, other):
        return np.dot(self, other)

    def is_orthog(self, other):
        return self.dot(other) == 0.0

    @staticmethod
    def are_collinear(*vectors):
        return np.linalg.matrix_rank(np.array([*vectors])) == 1

    @staticmethod
    def cross(self, other):
        return Vector(np.cross(self, other))

    def norm(self):
        return np.sqrt(np.sum(self._coord ** 2))

    def normalize(self):
        return self / self.norm()

    @staticmethod
    def angle(self, other):
        return np.arccos(self.dot(other) / (self.norm() * other.norm()))

    @staticmethod
    def projection(v, w):
        e = w.normalize()
        return e * v.dot(e)


