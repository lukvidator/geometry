from Point import Point
import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException


class Vector(Point):
    def __init__(self, *points, dtype=np.float64):
        """
        Create a vector.

        Parameters
        ----------
        point1: array_like
        point2: array_like
        dtype: data-type, optional

        Returns
        -------
        out : Vector
        """
        if len(points) == 1:
            self._coord = np.array(points[0], dtype)
        elif len(points) == 2:
            self._coord = np.array(points[1], dtype=dtype) - np.array(points[0], dtype=dtype)
        else:
            pass    # TODO: raise exception

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
        """
        Find the dot product of two vectors.

        Parameters
        ----------
        other : Vector

        Returns
        -------
        out : int
        """
        return np.dot(self, other)

    def is_orthog(self, other):
        """
        Find out if two vectors are orthogonal to each other.

        Parameters
        ----------
        other : Vector

        Returns
        -------
        out : bool
        """
        return self.dot(other) == 0.0

    @staticmethod
    def are_collinear(*vectors):
        """
        Find out if vectors are collinear to each other.

        Parameters
        ----------
        vectors : Vectors

        Returns
        -------
        out : bool
        """
        return np.linalg.matrix_rank(np.array([*vectors])) == 1

    @staticmethod
    def cross(self, other):
        """
        Find the cross product of two vectors.

        Parameters
        ----------
        self : Vector
        other : Vector

        Returns
        -------
        out : Vector
        """
        return Vector(np.cross(self, other))

    def norm(self):
        """
        Find the norm of vector.

        Returns
        -------
        out : float
        """
        return np.sqrt(np.sum(self._coord ** 2))

    def normalize(self):
        """
        Create normalized vector.

        Returns
        -------
        out : Vector
        """
        return self / self.norm()

    @staticmethod
    def angle(self, other):
        """
        Find the angle between two vectors.

        Parameters
        ----------
        self : Vector
        other : Vector

        Returns
        -------
        out : float
        """
        return np.arccos(self.dot(other) / (self.norm() * other.norm()))

    @staticmethod
    def projection(v, w):
        """
        Find the projection of vector v onto vector w.

        Parameters
        ----------
        v : Vector
        w : Vector

        Returns
        -------
        out : Vector
        """
        e = w.normalize()
        return e * v.dot(e)

    @staticmethod
    def direction_case(vector1, vector2) -> int:
        """
        Give the direction case of vectors.

        -1 -- opposite directions
        1 -- co-focus vectors
        0 -- otherwise

        Parameters
        ----------
        vector1 : Vector
        vector2 : Vector

        Returns
        -------
        out : int
        """
        if Vector.are_collinear(vector1, vector2):
            return 1 if vector1.normalize() == vector1.normalize() else -1
        else:
            return 0

