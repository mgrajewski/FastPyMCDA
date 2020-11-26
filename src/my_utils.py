# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:30:51 2019

@author: Grasse
"""

import itertools as it


def list_split(input_list, delimiter):
    #  TODO: add docstring
    grouped = (list(g) for k, g in it.groupby(input_list,
                                              key=lambda x: x in delimiter))
    filtered = list(it.filterfalse(lambda x: x[0] in delimiter, grouped))
    return filtered
