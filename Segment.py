import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Line import Line
from Point import Point
from tools import rectangle_test
from Vector import Vector
from matplotlib import pyplot as plt


class Segment:
    def __init__(self, points):
        """
        Create a segment.

        Parameters
        ----------
        points : array_like

        Returns
        -------
        out : Segment
        """
        self.points = points

    @classmethod
    def from_point_and_vector(cls, point, vector):
        """
        Create a segment.

        Parameters
        ----------
        point : Point
        vector : Vector

        Returns
        -------
        out : Segment
        """
        return cls((lambda p, v: [p, p + v])(Point(point), Vector(vector)))

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if len(points) == 2:
            self._points = [Point(point) for point in points]
        else:
            raise WrongTypeException(f"Can't set {type(self)} points with {points}")

    def __getitem__(self, item):
        return self._points[item]

    def __setitem__(self, key, value):
        self._points[key] = Point(value)

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self._points) + ")"

    __repr__ = __str__

    def __reversed__(self):
        return self._points.__reversed__()

    def reversed(self):
        return Segment(list(self.__reversed__()))

    def __eq__(self, other):
        if self is other:
            return True
        else:
            return self._points == other.points or list(self._points.__reversed__()) == other.points

    def __ne__(self, other):
        return not (self == other)

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
        return self._points[0] + t * Vector(*self._points)

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
        vectors = [Vector(self._points[0], point), Vector(*self._points)]
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
            return rectangle_test(self._points, point)
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
        q = Point(segment1[0] + t1 * v)

        return int(0 <= t1 <= 1 and 0 <= t2 <= 1), q, t1, t2

    def midpoint(self):
        return (self._points[0] + self._points[1]) / 2

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
        return ax.plot([self._points[0][0], self._points[1][0]], [self._points[0][1], self._points[1][1]], **kwargs)
