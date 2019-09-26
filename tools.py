from functools import reduce
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


def find_three_plane_points(coefs):
    # TODO: extend function to n-dim
    index = coefs.nonzero()[0][0]     # finding the axis with the first nonzero coefficient
    others = np.concatenate([coefs[:index], coefs[index + 1:]])    # and drop it
    others = -others / coefs[index]    # transferring other coefficients to another side and divide them by nonzero coef

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
