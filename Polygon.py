from typing import List
from Point import Point
from Line import Line
from Segment import Segment
from Vector import Vector
from numpy import sign, array
from numpy.linalg import det


class Polygon:
    def __init__(self, points: List[Point], name=''):
        self.name = name
        self.points = points

    @property
    def square(self):
        """
        :return: square of Polygon
        """
        result = 0
        for i in range(2, len(self.points)-1):
            result += det([
                array(self.points[0]) - array(self.points[i+1]),
                array(self.points[i+1]) - array(self.points[i]),
            ])

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


# p = Polygon([
#     Point([0, 0]),
#     Point([1, 0]),
#     Point([1, 1]),
#     Point([0, 1]),
# ])
#
# v = Vector([1, 1])
#
# l = Line(Point([0, 0]), Point([1, 0]))
#
# print(p.square)
# print(p.orientation)
# print(p.vertex_number)
# print(p.edges)
