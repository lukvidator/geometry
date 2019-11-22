import numpy as np


class Vector(np.ndarray):
    def __new__(cls, start_point, end_point=None, dtype=np.float, copy=True):
        if end_point is None:
            return np.array(start_point, dtype=dtype, copy=copy).view(cls)
        else:
            return (np.array(end_point, dtype=dtype, copy=copy) -
                    np.array(start_point, dtype=dtype, copy=copy)).view(cls)

    @classmethod
    def from_iter(cls, iterable, dtype=np.float, count=-1):
        return np.fromiter(iterable, dtype, count=count).view(cls)

    @classmethod
    def from_function(cls, func, length, dtype=np.float):
        return np.fromfunction(func, shape=(length,), dtype=dtype).view(cls)

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return bool(np.all(np.ndarray.__eq__(self, other)))

    def __ne__(self, other):
        return not self == other

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
        return np.linalg.matrix_rank(vectors) == 1

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

    def norm(self, norm="euclidean"):
        """
        Find the norm of vector.

        Returns
        -------
        out : float
        """
        return np.sqrt(self.dot(self))

    def normalize(self):
        """
        Create normalized vector.

        Returns
        -------
        out : Vector
        """
        return self / self.norm()

    @staticmethod
    def _angle2d_sign(self, other):
        d = np.linalg.det(np.array([self, other]))
        return np.sign(d) if not np.isclose(d, 0.) else 1

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
        sign = Vector._angle2d_sign(self, other) if len(self) == 2 else 1
        return sign * np.arccos(self.dot(other) / (self.norm() * other.norm()))

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
            return 1 if vector1.normalize() == vector2.normalize() else -1
        else:
            return 0

    def octant(self):
        if len(self) == 2:
            if np.count_nonzero(self) == 0:
                return 0

            x, y = self
            if 0 <= y < x:
                return 1
            elif 0 < x <= y:
                return 2
            elif -y < x <= 0:
                return 3
            elif 0 < y <= -x:
                return 4
            elif x < y <= 0:
                return 5
            elif y <= x < 0:
                return 6
            elif 0 <= x < -y:
                return 7
            else:
                return 8

    def __str__(self):
        return super(self.__class__, self).__repr__()
