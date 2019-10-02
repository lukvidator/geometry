from functools import reduce
from itertools import filterfalse
import numpy as np
import re


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


def _nf2(a, b, p):
    return np.linalg.det([np.array(p) - np.array(a), np.array(b) - np.array(a)])


def triangle_signed_square(a, b, c):
    return .5 * _nf2(a, c, b)


def rectangle_test(points, point):
    minmax = [np.amin(points), np.amax(points)]
    return True if (point >= minmax[0]).all() and (point <= minmax[1]).all() else False


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
