import numpy as np
from Exceptions import WrongTypeException, WrongDimensionException
from Point import Point
from Vector import Vector


class Segment:
    def __init__(self, *args):
        args = self._prepare_init_args(args)
        types = [type(arg) for arg in args]

        if types == [Point, Point]:
            self._points = args
        elif types == [Point, Vector]:
            self._points = [args[0], args[0] + args[1]]
        elif types == [np.ndarray, np.ndarray]:
            self._points = [Point(arg) for arg in args]
        else:
            raise WrongTypeException(f"Can't init {type(self)} with {args}")

    @staticmethod
    def _prepare_init_args(args):
        assert len(args) in (1, 2)
        args = args if len(args) == 2 else args[0]
        assert len(args) == 2

        return args

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if len(points) == 2 and [type(point) for point in points] == [Point, Point]:
            self._points = points
        else:
            raise WrongTypeException(f"Can't set {type(self)} points with {points}")

    # TODO: implement Segment class
    pass
