import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Plane import Plane
from Point import Point
from Vector import Vector


class Line:
    def __init__(self, point, vector):
        """
        Create a line.

        Parameters
        ----------
        point : Point
        vector : Vector

        Returns
        -------
        out : Line
        """
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
        """
        Create a line.

        Parameters
        ----------
        point1 : Point
        point2 : Point

        Returns
        -------
        out : Line
        """
        return cls(point1, Vector(point1, point2))

    @classmethod
    def from_segment(cls, segment):
        """
        Create a line.

        Parameters
        ----------
        segment : Segment

        Returns
        -------
        out : Line
        """
        return cls.from_points(segment[0], segment[-1])

    def points(self):
        """
        Line's points.

        Returns
        -------
        out : list
        """
        return [self._point, self._point + self._vector]

    def dim(self):
        """
        The dimension of the line's data.

        Returns
        -------
        out : int
        """
        return len(self._point)

    def coefficients(self):
        """
        Find line's equation coefficients.

        Returns
        -------
        out : np.ndarray
        """
        if self.dim() == 2:
            normal = Vector([-self._vector[-1], self._vector[0]])
            return np.array([*normal, -normal.dot(self._point)])
        else:
            raise WrongDimensionException(f"Can't get coefficients of the line in {self.dim()} dimension")

    def __call__(self, point):
        """
        Find the value of substituting the point into the line's equation.

        Parameters
        ----------
        point: Point

        Returns
        -------
        out : float
        """
        if self.dim() == 2:
            return np.dot(np.append(point, 1.), self.coefficients())
        else:
            raise WrongDimensionException(f"Can't evaluate Line's equation value in {self.dim()} dimension")

    def parameter(self, t):
        """
        Find a point according to the line data corresponding to parameter t.

        Parameters
        ----------
        t : float

        Returns
        -------
        out : Point
        """
        return self._point + t*self._vector

    def __str__(self):
        return self.__class__.__name__ + "(" + str(self._point) + ", " + str(self._vector) + ")"

    __repr__ = __str__

    def plot(self, ax, **kwargs):
        """
        Add line to the axes using kwargs.

        Parameters
        ----------
        ax : Axes or Axes3D (matplotlib)
        kwargs : kwargs from LineCollection (matplotlib)

        Returns
        -------
        out : Line2DCollection in case of Axes or Line3DCollection in case of Axes3D
        """
        index = self._vector.nonzero()[0][0]    # finding index of nonzero self.vector coordinate
        lim = np.array([ax.get_xlim, ax.get_ylim, ax.get_zlim][index]())    # choosing the corresponding lim
        t = (lim - self._point[index]) / self._vector[index]    # evaluating the parameter t for the param line equation
        points = np.array([self.parameter(param) for param in t])    # finding points for plotting line
        xyz = zip(*points)
        return ax.plot(*xyz)

    def distance_to_point(self, point):
        """
        Find absolute distance from line to point

        Parameters
        ----------
        point : Point

        Returns
        -------
        out : float
        """
        return np.abs(np.linalg.det(np.array([point - self._point, self._vector])))

    @staticmethod
    def relation(line1, line2) -> (0, 1, 2, 3):
        """
        Find relation between lines.

        - 0 if lines are intersected,
        - 1 if lines are equal
        - 2 if lines are parallel
        - 3 if lines are intercrossed

        Parameters
        ----------
        line1 : Line
        line2 : Line

        Returns
        -------
        out : int
        """
        plane1 = Plane(line1.point, [line1.vector])
        plane2 = Plane(line2.point, [line2.vector])
        case = Plane.relation(plane1, plane2)
        return case if not case else case - 1

    # TODO: implement Line.from_planes
