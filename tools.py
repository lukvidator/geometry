from functools import reduce
from operator import itemgetter
from itertools import filterfalse, chain
import numpy as np
import re
from Vector import Vector


def extract_coefs(equation, variables):
    assert type(variables) in (str, list, tuple)
    assert isinstance(equation, str)
    assert len(variables) > 0

    equation = equation.translate(str.maketrans({" ": ""}))  # deleting the whitespaces
    pattern = r"(?P<sign>[-+]?)(?P<coef>\d+(?:\.\d*)?|\.\d+)\*(?P<x>" + "|".join(variables) + ")"

    pattern_compiled = re.compile(pattern)
    matches = pattern_compiled.findall(equation)

    variables_found = {x[-1] for x in matches}

    if len(variables_found) != len(variables):  # if there are any missing variables
        variables_to_insert = set(variables).difference(variables_found)  # finding missing ones

        for x in variables_to_insert:
            matches.append(('', '0', x))

    def sort_criteria(item):
        nonlocal variables
        return variables.index(item[-1])

    matches.sort(key=sort_criteria)

    # for the last coefficient (free one)
    # removing all the terms except the free one
    deleting = [match[0] + match[1] + "*" + match[-1] for match in matches]
    equation = reduce(lambda res, to_delete: res.replace(to_delete, ""), deleting, equation)

    pattern = r"(?P<sign>[-+]?)(?P<coef>\d+(?:\.\d*)?|\.\d+)[^\*\.\d]"
    pattern_compiled = re.compile(pattern)

    match = pattern_compiled.findall(equation)
    if not match:
        match = [('', '0')]

    matches += match

    return np.array([float(x[0] + x[1]) for x in matches])


def find_three_plane_points(coefficients):
    # TODO: extend function to n-dim
    index = coefficients.nonzero()[0][0]     # finding the axis with the first nonzero coefficient
    others = np.concatenate([coefficients[:index], coefficients[index + 1:]])    # and drop it
    # transferring other coefficients to another side and divide them by nonzero coefficient
    others = -others / coefficients[index]

    instances = np.zeros([3, len(others) - 1])    # we'll substitute 3 different points
    instances[0, 0] = 1.
    instances[1, 1] = 1.
    instances = np.hstack([instances, np.ones([3, 1])])

    c = np.array([[np.dot(others, inst)] for inst in instances])    # finding the values of "nonzero" axis

    instances = instances[:, :-1]    # removing the last column of free coefficients
    if index == 0:    # forming the result points
        instances = np.hstack([c, instances])
    elif index == len(others) - 1:
        instances = np.hstack([instances, c])
    else:
        instances = np.split(instances, index, axis=1)
        instances.insert(index, c)
        instances = np.hstack(instances)

    return instances


def rotation_matrix(phi, w=None):
    if not w:
        return np.array([[np.cos(phi), -np.sin(phi)],
                         [np.sin(phi), np.cos(phi)]])
    else:
        pass    # TODO: implement rotation_matrix for 3D


def nf2(a, b, p):
    """
    Returns the result of f(p), where
    f is the equation of the line based on
    the vector (b - a) with the start point as a.

    Parameters
    ----------
    a, b, p : ndarray or Point

    Returns
    -------
    out : float
        out is f(p) and represents:
        0 - if p is on the line,
        > 0 - if p is on the right side,
        < 0 - if p is on the left side.
    """
    return np.linalg.det([np.array(p) - np.array(a), np.array(b) - np.array(a)])


def triangle_signed_square(a, b, c):
    return .5 * nf2(a, c, b)


def rectangle_test(points, point):
    """
    Find out if the point inside the rectangle, which is formed by min/max coords of the Polygon's points.

    Parameters
    ----------
    points: array-like
        Points should be the matrix with shape (Np, dim).
    point : array-like
        Point should be an array-like object with length == dim.

    Returns
    -------
    out : bool
    """
    minmax = [np.amin(points), np.amax(points)]
    return True if (np.array(point) >= minmax[0]).all() and (np.array(point) <= minmax[1]).all() else False


def form_contours(segments):
    """
    Return the contours formed from the list of segments.

    Parameters
    ----------
    segments : list(Segments)

    Returns
    -------
    out : list
        [] -- if segments is empty
        [[...], [...], ...] -- if segments can form the contours
    """
    contours = []
    if not segments:
        return contours

    k = -1
    # duplicates?

    while segments:  # while segments is not empty
        k += 1  # move to the next contour
        contours.append([])  # append the new list for the current contour
        contours[k].append(segments.pop(0))  # and take the first segment from segments as the start segment

        while contours[k][0][0] != contours[k][-1][-1]:  # while current contour is not closed
            is_appended = False

            for i, segment in enumerate(segments):
                # if the start of the segment is the current endpoint of the contour
                if contours[k][-1][-1] == segment[0]:
                    contours[k].append(segments.pop(i))
                    is_appended = True
                    break
                elif contours[k][-1][-1] == segment[1]:
                    contours[k].append(segments.pop(i).reversed())
                    is_appended = True
                    break

            # if for-loop don't found the next segment to form contour
            if not is_appended:
                raise ValueError("Can't form contour: there is no any segment to continue forming contour")
    # TODO: improve function in case of the common point on the bound of the window (p. 430 at the bottom)
    return contours


def unique_everseen(iterable, key=None):
    """
    List unique elements, preserving order. Remember all elements ever seen.
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D

    Parameters
    ----------
    iterable : iterable
    key : function, default: None

    Returns
    -------
    out : iter
    """
    seen = list()
    seen_add = seen.append

    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def append(iterable, item):
    return chain(iterable, [item])


def find_min_point(points):
    return np.min(points[points[:, 1] == np.min(points[:, 1])], axis=0)


def convex_hull(points):
    ch = [find_min_point(points)]

    while True:
        ch.append(ch[-1])

        for point in points:
            square = triangle_signed_square(ch[-2], point, ch[-1])
            if square > 0. or (square == 0. and Vector(ch[-2], ch[-1]).norm() < Vector(ch[-2], point).norm()):
                ch[-1] = point

        if np.allclose(ch[0], ch[-1]):
            break

    return ch


def angle(point):
    x, y = point[0], point[1]
    if x > 0:
        if y >= 0:
            return np.arctan(y / x)
        else:
            return np.arctan(y / x) + 2 * np.pi
    elif x < 0:
        return np.arctan(y / x) + np.pi
    elif x == 0.:
        if y > 0:
            return np.pi / 2
        elif y < 0:
            return 3 * np.pi / 2
        else:
            return np.nan


def polar(point):
    return np.array([angle(point), np.dot(point, point)])


def polar_sort_points(points, min_point):
    points = np.hstack([points, np.array([polar(point) for point in points - min_point])])
    return np.array(sorted(points, key=itemgetter(2, 3)))[:, :2]


def graham_convex_hull(points):
    ch = [find_min_point(points)]
    points = points[[row.any() for row in ~(points == ch[0])]]
    points = polar_sort_points(points, ch[0])

    for point in points:
        while len(ch) > 1 and triangle_signed_square(ch[-2], ch[-1], point) <= 0:
            ch.pop()

        ch.append(point)

    return np.array(ch)
