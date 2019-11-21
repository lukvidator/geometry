import numpy as np
from Vector import Vector
from matplotlib import pyplot as plt


class Point(Vector):
    def __new__(cls, point, dtype=np.float, copy=True):
        """
        Create a Point.

        Parameters
        ----------
        point : array_like
            An array, any object exposing the array interface, an object whose
            __array__ method returns an array, or any (nested) sequence.
        dtype : data-type, optional
            The desired data-type for the Point.  If not given, then the type will
            be float.
        copy : bool, optional
            Used to specify if the point should be copied or not.

        Returns
        -------
        out : Point
        """
        return super().__new__(cls, point, dtype=dtype, copy=copy)

    __array_priority__ = 0.1

    def plot(self, ax=None, **kwargs):
        """
        Add point to the axes using kwargs.

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
        return ax.scatter(*self, **kwargs)
