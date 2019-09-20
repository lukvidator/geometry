import numpy as np
import unittest
from Exceptions import WrongDimensionException, WrongTypeException
from Plane import Plane
from Point import Point
from tools import extract_coefs, find_three_plane_points
from Vector import Vector


class PointTestCase(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(Point([1, 1, 1]), Point)

    def test_coord(self):
        self.assertIsInstance(Point([1, 1, 1]).coord, np.ndarray)

    def test_eq(self):
        self.assertEqual(Point([1, 1, 1]), Point([1, 1, 1]))
        self.assertRaises(WrongTypeException, Point([1, 1, 1]).__eq__, [1, 1, 1])
        self.assertRaises(WrongDimensionException, Point([1, 1, 1]).__eq__, Point([1, 1]))

    def test_ne(self):
        self.assertNotEqual(Point([1, 1, 1]), Point([2, 2, 2]))

    def test_add(self):
        self.assertRaises(WrongTypeException, Point([1, 1]).__add__, [1, 1])
        self.assertEqual(Point([1, 1, 1]) + Point([2, 2, 2]), Point([3, 3, 3]))

    def test_len(self):
        self.assertEqual(len(Point([1, 1, 1])), 3)
        self.assertEqual(len(Point([])), 0)

    def test_getitem(self):
        for i in range(3):
            self.assertEqual(Point([0, 1, 2])[i], i)
        self.assertRaises(IndexError, Point([1, 2, 3]).__getitem__, 3)

    def test_iter(self):
        for x in Point([1, 1, 1]):
            self.assertIsInstance(x, float)

    def test_contains(self):
        self.assertTrue(1 in Point([1, 2, 3]))
        self.assertFalse(4 in Point([1, 2, 3]))

    def test_str(self):
        self.assertEqual(str(Point([1, 2])), "Point([1., 2.])")

    def test_repr(self):
        self.assertEqual(repr(Point([1, 2])), "Point([1., 2.])")

    def test_setitem(self):
        p = Point([0, 2, 3])
        p[0] = 1
        self.assertEqual(Point([1, 2, 3]), p)

    def test_getattr(self):
        p = Point([2, 3, 1])
        p.sort()
        self.assertEqual(Point([1, 2, 3]), p)


class VectorTestCase(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(Vector(Point([1, 1, 1])), Vector)
        self.assertIsInstance(Vector(Point([0, 0, 0]), Point([1, 1, 1])), Vector)
        self.assertIsInstance(Vector(np.array([1, 1, 1])), Vector)
        self.assertIsInstance(Vector(np.array([0, 0, 0]), np.array([1, 1, 1])), Vector)
        self.assertRaises(WrongTypeException, Vector.__init__, Vector.__new__(Vector), [1, 1, 1], [1, 0, 0], [0, 0, 1])

    def test_eq(self):
        self.assertEqual(Vector(Point([1, 1, 1])), Vector(np.array([1, 1, 1])))

    def test_neg(self):
        self.assertEqual(-Vector(Point([1, 1, 1])), Vector([-1, -1, -1]))

    def test_add(self):
        self.assertEqual(Vector([3, 3, 3]), Vector([1, 1, 1]) + Vector([2, 2, 2]))
        self.assertRaises(WrongTypeException, Vector([1, 1, 1]).__add__, [1, 1, 1])
        self.assertRaises(WrongDimensionException, Vector([1, 1, 1]).__add__, Vector([1, 1]))

    def test_mul(self):
        self.assertEqual(Vector([2, 2, 2]), Vector([1, 1, 1]) * 2)

    def test_truediv(self):
        self.assertEqual(Vector([1, 1, 1]), Vector([2, 2, 2]) / 2)

    def test_cross(self):
        self.assertEqual(Vector.cross(Vector([1, 0, 0]), Vector([0, 1, 0])), Vector([0, 0, 1]))

    def test_dot(self):
        self.assertEqual(Vector.dot(Vector([1, 1, 1]), Vector([1, 1, 1])), 3)
        self.assertEqual(Vector.dot(Vector([1, 0, 0]), Vector([0, 1, 0])), 0)

    def test_str(self):
        self.assertEqual(str(Vector([1, 1])), "Vector([1., 1.])")

    def test_repr(self):
        self.assertEqual(repr(Vector([1, 1])), "Vector([1., 1.])")

    def test_norm(self):
        self.assertEqual(Vector([1, 0, 0]).norm(), 1)
        self.assertEqual(Vector([1, 1]).norm(), np.sqrt(2))

    def test_normalize(self):
        self.assertEqual(Vector([1, 0, 0]), Vector([3, 0, 0]).normalize())

    def test_angle(self):
        self.assertAlmostEqual(Vector([1, 0, 0]).angle(Vector([0, 1, 0])), np.pi/2)


class PlaneTestCase(unittest.TestCase):
    def test_init(self):
        self.assertIsInstance(Plane(np.array([1, 1, 1, 1])), Plane)
        self.assertIsInstance(Plane(Point([0, 0, 0]), Vector([1, 0, 0])), Plane)
        self.assertIsInstance(Plane(np.array([0, 0, 0]), np.array([1, 0, 0])), Plane)
        self.assertIsInstance(Plane(Point([0, 0, 0]), Vector([0, 1, 0]), Vector([0, 0, 1])), Plane)
        self.assertIsInstance(Plane(Point([0, 0, 0]), Point([0, 1, 0]), Point([0, 0, 1])), Plane)
        self.assertRaises(WrongTypeException, Plane.__init__, Plane.__new__(Plane), np.array([]), np.array([]), np.array([]))
        self.assertRaises(WrongTypeException, Plane.__init__, Plane.__new__(Plane), 2)

    def test_coefficients(self):
        self.assertTrue((Plane(np.array([1, 1, 1, 1])).coefficients == np.array([1, 1, 1, 1])).all())


class ToolsTestCase(unittest.TestCase):
    def test_extract_coefs(self):
        self.assertTrue((extract_coefs("-5.1*x + 4*y - 1.9*z - 1 == 0", "xyz") == np.array([-5.1, 4., -1.9, -1.0])).all())
        self.assertTrue((extract_coefs("1.9*z + 4*y == 0", "xyz") == np.array([0., 4., 1.9, 0.])).all())
        self.assertTrue((extract_coefs("-5.1*x1 + 4*x2 - 1.9*x3 - 1 == 0", ["x1", "x2", "x3"]) == np.array([-5.1, 4., -1.9, -1.0])).all())
        self.assertTrue((extract_coefs("1.9*x3 + 4*x2 == 0", ["x1", "x2", "x3"]) == np.array([0., 4., 1.9, 0.])).all())


# TODO: more tests

if __name__ == '__main__':
    unittest.main()
