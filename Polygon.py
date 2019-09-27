from typing import List
from Point import Point
from Segment import Segment
from numpy import sign, array, amin, amax
from numpy.linalg import det


class Polygon:
    def __init__(self, points: List[Point]):
        self.points = points

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = [Point(point) for point in points]

    @staticmethod
    def _nf2(a: Point, b: Point, p: Point):
        return det([
            array(p) - array(a),
            array(b) - array(a)
        ])

    @property
    def square(self):
        """
        :return: sign square of Polygon
        """
        result = 0
        for i in range(1, len(self.points)-1):
            result += 0.5*self._nf2(self.points[0], self.points[i+1], self.points[i])

        return result

    @property
    def orientation(self):
        """
        :return:
        0 - if orientation is left
        1 - if orientation is right
        """
        return 1 if self.square > 0 else 0

    @property
    def vertex_number(self):
        """
        :return: number of vertex
        """
        return len(self.points)

    @property
    def edges(self):
        n = len(self.points)
        result = [Segment([self.points[i], self.points[i+1]]) for i in range(n-1)]
        result.append(Segment([self.points[n - 1], self.points[0]]))
        return result

    @property
    def is_convex(self):
        """
        :return:
        0 - if it's not convex
        1 - if it's convex
        """
        n = len(self.points) - 1
        begin_orient = self._nf2(self.points[n], self.points[0], self.points[1])
        for i in range(1, n):
            current_orient = self._nf2(self.points[i-1], self.points[i], self.points[i+1])
            if sign(begin_orient) != sign(current_orient):
                break
        else:
            last_orient = self._nf2(self.points[n-1], self.points[n], self.points[0])
            if sign(begin_orient) == sign(last_orient):
                return 1
        return 0

    def rectangle_test(self, point):
        minmax = [amin(self._points), amax(self._points)]
        return True if (point >= minmax[0]).all() and (point <= minmax[1]).all() else False

# p = Polygon([
#     Point([0, 0]),
#     Point([2, 0]),
#     Point([2, 2]),
#     Point([0, 2]),
# ])
#
# print(p.square)
# print(p.orientation)
# print(p.vertex_number)
# print(p.edges)
# print(p.is_convex)
#
# p2 = Polygon([
#     Point([0, 0]),
#     Point([2, 1]),
#     Point([0, 2]),
#     Point([1, 1])
# ])
#
# print(p2.square)
# print(p2.orientation)
# print(p2.is_convex)
