import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Line import Line
from Point import Point
from tools import rectangle_test
from Vector import Vector
from matplotlib import pyplot as plt
# import itertools as it
import more_itertools as mit


class Segment(np.ndarray):
    def __new__(cls, points, dtype=np.float, copy=True):
        """
        Create a segment.

        Parameters
        ----------
        points : array_like

        Returns
        -------
        out : Segment
        """
        return np.array(points, dtype=dtype, copy=copy).view(cls)

    @classmethod
    def from_point_and_vector(cls, point, vector, dtype=np.float, copy=True):
        """
        Create a segment.

        Parameters
        ----------
        point : Point
        vector : Vector
        dtype : data-type, optional
            Any object that can be interpreted as a numpy data type.
        copy : bool, optional
            If true (default), then the object is copied.  Otherwise, a copy will
            only be made if __array__ returns a copy, if obj is a nested sequence,
            or if a copy is needed to satisfy any of the other requirements
            (`dtype`, `order`, etc.).

        Returns
        -------
        out : Segment
        """
        return cls((lambda p, v: [p, p + v])(Point(point), Vector(vector)), dtype=dtype, copy=copy)

    @classmethod
    def from_iter(cls, iterable, dtype=np.float, copy=True):
        points = mit.take(2, iterable)
        return cls(points, dtype=dtype, copy=copy)

    @property
    def points(self):
        return self

    def __getitem__(self, item):
        return np.ndarray.__getitem__(self, item).view(Point)

    def __str__(self):
        return super(self.__class__, self).__repr__()

    def reversed(self):
        return self.__class__.from_iter(reversed(self))

    def __eq__(self, other):
        if self is other:
            return True
        else:
            eq = np.ndarray.__eq__
            result = np.all(eq(self, other)) or np.all(eq(self.reversed(), other))
            return bool(result)

    def __ne__(self, other):
        return not self == other

    def point_by_parameter(self, t):
        """
        Find a point according to the segment corresponding to parameter t.

        Parameters
        ----------
        t : float

        Returns
        -------
        out : Point
        """
        return (self[0] + t * self[1]).view(Point)

    def parameter_by_point(self, point):
        """
        Find a parameter according to the segment corresponding to the point.

        Parameters
        ----------
        point : Point

        Returns
        -------
        out : float
        """
        vectors = [Vector(self[0], point), Vector(*self)]
        if Vector.are_collinear(*vectors):
            return Vector.direction_case(*vectors) * vectors[0].norm() / vectors[1].norm()
        else:
            raise ValueError("Can't get parameter of a point, which is not on the segment")

    @staticmethod
    def relation(segment1, segment2) -> (-1, 0, 1):
        """
        Find relation between segments.

        1 -- if segments are intersected,
        0 -- if segments aren't parallel and not intersected,
        -1 -- otherwise

        Parameters
        ----------
        segment1: Segment
        segment2: Segment

        Returns
        -------
        out : int
        """
        # incorrect
        # TODO: tests
        line1 = Line.from_segment(segment1)
        case = Line.relation(line1, Line.from_segment(segment2))
        if case == 0:
            return 1 if line1(segment2[0]) * line1(segment2[1]) <= 0 else 0
        else:
            return -1

    @staticmethod
    def _are_intersected(segment1, segment2):
        d1 = np.linalg.det([segment1[1] - segment1[0], segment2[0] - segment1[0]])
        d2 = np.linalg.det([segment1[1] - segment1[0], segment2[1] - segment1[0]])
        d3 = np.linalg.det([segment2[1] - segment2[0], segment1[0] - segment2[0]])
        d4 = np.linalg.det([segment2[1] - segment2[0], segment1[1] - segment2[0]])
        return d1 * d2 <= 0 and d3 * d4 <= 0

    def _is_point_on_segment(self, point: Point) -> bool:
        if Vector.are_collinear(Vector(self[0], point), Vector(self[0], self[1])):
            return rectangle_test(self, point)
        else:
            return False

    @staticmethod
    def intersection(segment1, segment2):
        """
        Find the intersection of two segments.

        (-1, None, None, None) -- if segments are parallel.
        (0, q, t1, t2) -- if segments are non-parallel, but don't have intersection.
                            There q is the intersection point of the segments lines and
                            t1, t2 are the parameters for this point in their parametrization.
        (1, q, t1, t2) -- if segments have intersection.

        Parameters
        ----------
        segment1 : Segment
        segment2 : Segment

        Returns
        -------
        out : tuple(int, Point, float, float)
        """
        v = Vector(*segment1)
        m = np.array([v, Vector(segment2[1], segment2[0])])

        if np.linalg.det(m) == 0:
            return -1, None, None, None

        t1, t2 = np.dot(Vector(segment1[0], segment2[0]), np.linalg.inv(m))
        q = segment1[0] + t1 * v

        return int(0 <= t1 <= 1 and 0 <= t2 <= 1), q, t1, t2

    def midpoint(self):
        return sum(self) / 2

    def plot(self, ax=None, **kwargs):
        """
        Add segment to the axes using kwargs.

        Parameters
        ----------
        ax : Axes or Axes3D (matplotlib)
        kwargs : kwargs from LineCollection (matplotlib)

        Returns
        -------
        out : Line2DCollection in case of Axes or Line3DCollection in case of Axes3D
        """
        if ax is None:
            ax = plt.gca()
        return ax.plot(self[:, 0], self[:, 1], **kwargs)
