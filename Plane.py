import numpy as np
from Exceptions import WrongDimensionException, WrongTypeException
from Point import Point
from tools import extract_coefs, find_three_plane_points
from Vector import Vector


class Plane:
    def __init__(self, point, vectors):
        """
        Create a plane.

        Parameters
        ----------
        point : array_like
        vectors : array_like

        Returns
        -------
        out : Plane
        """
        if len(point) == len(vectors[0]):
            self.point = point
            self.vectors = vectors
        else:
            raise WrongDimensionException(f"Can't init {self.__class__.__name__} with args of different dimensions")

    @classmethod
    def from_coefficients(cls, coefficients):
        """
        Create a plane.

        Parameters
        ----------
        coefficients : np.ndarray

        Returns
        -------
        out : Plane
        """
        return cls.from_points(find_three_plane_points(coefficients))

    @classmethod
    def from_equation(cls, equation, variables):
        """
        Create a plane.

        Parameters
        ----------
        equation : str
        variables : array_like
            Array of string variables.

        Returns
        -------
        out : Plane
        """
        return cls.from_coefficients(extract_coefs(equation, variables))

    @classmethod
    def from_points(cls, points):
        """
        Create a plane.

        Parameters
        ----------
        points : array_like

        Returns
        -------
        out : Plane
        """
        return cls(points[0], [Vector(points[0], point) for point in points[1:]])

    @classmethod
    def from_point_and_normal(cls, point, normal):
        """
        Create a plane.

        Parameters
        ----------
        point : Point
        normal : Vector

        Returns
        -------
        out : Plane
        """
        if isinstance(point, Point) and isinstance(normal, Vector):
            return cls.from_coefficients(np.append(normal, -np.dot(point, normal)))
        else:
            raise WrongTypeException(f"Can't create {cls.__name__} with {(point, normal)}")

    def coefficients(self):
        """
        Find the coefficients of the plane's equation.

        Returns
        -------
        out : np.ndarray
        """
        matrix = np.array([self._point, *self._vectors])    # creating main matrix
        dim = self.dim()    # getting the space dimension

        coefficients = np.array(
            [(-1 if i % 2 else 1) * np.linalg.det(
                matrix[np.ix_(range(1, dim), [j for j in range(0, dim) if j != i])]
            ) for i in range(0, dim)])    # creating equation coefficients

        coefficients = np.append(coefficients, -np.linalg.det(matrix))    # adding free one

        return coefficients

    def equation(self, var=None):
        """
        Build the equation according to the plane's coefficients.

        Parameters
        ----------
        var : array_like
            Array of string variables.

        Returns
        -------
        out : str
        """
        coefficients = self.coefficients()
        if not var:
            var = ["x" + str(i + 1) for i in range(0, len(coefficients) - 1)]
        elif not len(var) == len(coefficients) - 1:
            raise WrongTypeException(f"Can't create equation using {var}")

        equation = \
            ' + '.join([str(pair[0]) + "*" + pair[1] for pair in zip(coefficients, var) if pair[0] != 0.])
        equation += " + " + str(coefficients[-1]) if coefficients[-1] != 0. else ""
        equation = equation.replace("+ -", "- ")
        equation += " == 0"

        return equation

    def normal(self):
        """
        Find the normal vector for the plane.

        Returns
        -------
        out : Vector
        """
        return Vector(self.coefficients()[:-1])

    def dim(self):
        """
        Find dimension of Plane's data.

        Returns
        -------
        out : int
        """
        return len(self._point)

    @property
    def vectors(self):
        return self._vectors

    @vectors.setter
    def vectors(self, vectors):
        if len(np.array(vectors).shape) == 2:
            self._vectors = [Vector(vector).normalize() for vector in vectors]
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
        """
        Find angle between planes.

        Parameters
        ----------
        plane1 : Plane
        plane2 : Plane

        Returns
        -------
        out : float
        """
        normal1 = plane1.normal()
        normal2 = plane2.normal()
        angle = Vector.angle(normal1, normal2)
        return angle if angle <= np.pi/2 else Vector.angle(-normal1, normal2)

    @staticmethod
    def _relation_cases(ranks) -> (0, 1, 2, 3, 4):
        if ranks[2] == ranks[3]:
            if ranks[1] == ranks[2]:
                if ranks[0] != ranks[1]:
                    return 1  # pl1 in pl2
                else:
                    return 2  # pl1 = pl2
            else:
                return 0  # pl1 intersect pl2
        else:
            if ranks[2] == ranks[1] and ranks[1] >= ranks[0]:
                return 3  # pl1 || pl2
            elif ranks[2] > ranks[1]:
                return 4  # pl1 and pl2 are intercrossed

    @staticmethod
    def relation(plane1, plane2):
        """
        Find relation of two planes

        0 -- plane1 intersect plane2
        1 -- plane1 in plane2
        2 -- plane1 = plane2
        3 -- plane1 || plane2
        4 -- plane1 and plane2 are intercrossed

        Parameters
        ----------
        plane1 : Plane
        plane2 : Plane

        Returns
        -------
        out : int
        """
        bridge = Vector(plane1.point, plane2.point)    # creating "bridge" vector
        vectors = np.vstack([plane1.vectors, plane2.vectors])

        ranks = [np.linalg.matrix_rank(plane.vectors) for plane in (plane1, plane2)]
        ranks.sort(key=len)
        ranks.extend([np.linalg.matrix_rank(system) for system in (vectors, np.append(vectors, bridge))])

        return Plane._relation_cases(ranks)

    def plot(self, ax, **kwargs):
        """
        Add plane's plot to the axes.

        Parameters
        ----------
        ax : Axes3D (matplotlib)
        kwargs : kwargs from Poly3DCollection (matplotlib)

        Returns
        -------
        out : Poly3DCollection
        """
        coefficients = self.coefficients()
        index = coefficients.nonzero()[0][0]

        axis_lims = [ax.get_xlim, ax.get_ylim, ax.get_zlim]
        axis_lims.pop(index)
        axis_lims = [lim() for lim in axis_lims]

        x = np.linspace(*axis_lims[0], 10)
        y = np.linspace(*axis_lims[1], 10)
        data = np.meshgrid(x, y)

        func_coefficients = -np.delete(coefficients, index) / coefficients[index]
        z = (lambda a, b: func_coefficients[0] * a + func_coefficients[1] * b + func_coefficients[2])(*data)

        data.insert(index, z)
        return ax.plot_surface(*data, **kwargs)
