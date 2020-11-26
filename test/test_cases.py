"""
@author: Simon Grasse, Forschungszentrum Juelich
Matthias Grajewski, FH Aachen University of Applied Sciences

This file contains a collection of regression tests. To run these, proceed as
follows:
    
1) Install the python package pytest by e.g. typing 'pip install pytest' in
   a shell/terminal (if not already installed)
2) Install the python package arraydiff by e.g. typing
   'pip install pytest-arraydiff' in a shell/terminal (if not already
   installed)
3) In a terminal (NOT a python interactive shell) switch to the directory
   'tests' within pymcda.
4) Start the tests by typing
   'py.test --arraydiff'
      
The tests compare the outcome of certain computations with our model with a
reference solution. These reference solutions are stored as txt-files in the
folder 'refsols'. If the current result and the reference result match, the
test is considered successful.
       
"""

import sys
import os

# python test package
import pytest
import numpy as np

# make sure root package-directory is in python search-path
if '__file__' in globals():
    ABS_DIR_NAME = os.path.abspath(os.path.dirname(__file__))
else:
    ABS_DIR_NAME = os.getcwd()
    
if os.path.abspath(os.path.join(ABS_DIR_NAME, '..', '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(ABS_DIR_NAME, '..', '..')))


from pymcda.src import wrapper as wr


@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_vec1():
    """
    Test the model for vectorial input data, default weights from Excel file
    """
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Ecological'
    arg2_name = 'Economic'
    
    # create a grid of evaluation points in the 'Ecological'-'Economic'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    # compute the classes for the given list of points (the model returns a
    # python list)
    result = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    # for using arraydiff, we need however to convert the output to a numpy array
    result = np.array(result, dtype=int)
    return result

@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_vec2():
    """
    Test the model for vectorial input data, default weights from Excel file
    """
    
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Social'
    arg2_name = 'Comfort'
    
    # create a grid of evaluation points in the 'Social'-'Comfort'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Social', pars2 for the values of 'Comfort')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    # compute the classes for the given list of points (the model returns a
    # python list)
    result = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    # for using arraydiff, we need however to convert the output to a numpy array
    result = np.array(result, dtype=int)
    return result

@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_compare_vec_scal_1():

    """
    Test the model for scalar input data and compare with the results for
    vectorial data (difference must be zero).
    The weights are the default ones from the Excel file.
    """
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Ecological'
    arg2_name = 'Economic'
    
    # create a grid of evaluation points in the 'Social'-'Comfort'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    numberOfRuns = len(pars1)
    resultsScal = range(0, numberOfRuns)
    resultsScal = list()
    
    # create results for scalar dat (one model run for each evaluation point)
    for i in range(0, numberOfRuns):
        aux_result = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1[i], pars2[i]])
        resultsScal.append(aux_result)

    resultsScal = np.array(resultsScal, dtype=int)
    
    # same for vectorial data (one model run for all evaluation points)
    resultsVec = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])

    # difference is hopefully zero
    result = resultsScal.flatten() - np.array(resultsVec, dtype=int)
    return result

@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_compare_vec_scal_2():

    """
    Test the model for scalar input data and compare with the results for
    vectorial data (difference must be zero).
    The weights are the default ones from the Excel file.
    """
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Social'
    arg2_name = 'Comfort'
    
    # create a grid of evaluation points in the 'Social'-'Comfort'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    numberOfRuns = len(pars1)
    resultsScal = range(0, numberOfRuns)
    resultsScal = list()
    
    # create results for scalar dat (one model run for each evaluation point)
    for i in range(0, numberOfRuns):
        aux_result = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1[i], pars2[i]])
        resultsScal.append(aux_result)

    resultsScal = np.array(resultsScal, dtype=int)
    
    # same for vectorial data (one model run for all evaluation points)
    resultsVec = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])

    # difference is hopefully zero
    result = resultsScal.flatten() - np.array(resultsVec, dtype=int)
    return result

@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_change_weights_1():
    """
    Test the model for vectorial input data. We however change weights
    beforehand.
    """
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Ecological'
    arg2_name = 'Economic'
    
    MyModel.set_weights_of_categories(['Social', 'Comfort'], [1, 1])
    
    
    # create a grid of evaluation points in the 'Ecological'-'Economic'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    # compute the classes for the given list of points (the model returns a
    # python list)
    result = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    # for using arraydiff, we need however to convert the output to a numpy array
    result = np.array(result, dtype=int)
    return result

@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_change_weights_2():
    """
    Test the model for vectorial input data. We however change weights
    beforehand.
    """
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Ecological'
    arg2_name = 'Comfort'
    
    MyModel.set_weights_of_categories(['Social', 'Economical'], [1, 1])
    
    
    # create a grid of evaluation points in the 'Ecological'-'Economic'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    # compute the classes for the given list of points (the model returns a
    # python list)
    result = MyModel.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    # for using arraydiff, we need however to convert the output to a numpy array
    result = np.array(result, dtype=int)
    return result

@pytest.mark.array_compare(file_format='text', reference_dir='refsols')
def test_change_weights_back_1():
    """
    Test the model for vectorial input data. We however change weights
    beforehand and then back again. This must not cause any effect.
    """
    
    MyModelDefault = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    MyModelChanged = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Ecological'
    arg2_name = 'Social'
    
    # both instances will produce the same results
    MyModelDefault.set_weights_of_categories(['Economic', 'Comfort'], [1, 2])
    MyModelChanged.set_weights_of_categories(['Economic', 'Comfort'], [1, 2])
    
    
    # create a grid of evaluation points in the 'Ecological'-'Economic'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = np.arange(0, 10.3, 0.5)
    pars2 = np.arange(0, 10.3, 0.5)
    
    pars1, pars2 = np.meshgrid(pars1, pars2)
    pars1 = pars1.flatten()
    pars2 = pars2.flatten()
    
    pars1 = pars1.tolist()
    pars2 = pars2.tolist()
    
    # results for the model with default weights (apart of Ecological and Economic)
    resultDefault = MyModelDefault.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    resultChanged = MyModelDefault.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])

    #change weights and compute again
    MyModelChanged.set_weights_of_categories(['Economic', 'Comfort'], [9, 6])
    resultChanged = MyModelDefault.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    #change weights back and compute again
    MyModelChanged.set_weights_of_categories(['Economic', 'Comfort'], [1, 2])
    resultChanged = MyModelDefault.get_class_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    # for using arraydiff, we need however to convert the output to a numpy array
    resultChanged = np.array(resultChanged, dtype=int)
    resultDefault = np.array(resultDefault, dtype=int)

    # difference must be zero
    result = resultChanged - resultDefault
    return result


@pytest.mark.array_compare(file_format='text', reference_dir='refsols', rtol=1)
def test_scal_scores1():
    """
    Test the model for scalar input data, default weights from Excel file
    """
    MyModel = wr.Wrapper('model_parameters_test.xlsx', 'car_users')
    arg1_name = 'Ecological'
    arg2_name = 'Economic'
    
    # create a grid of evaluation points in the 'Ecological'-'Economic'-plane
    # and convert the grid to two python lists (pars1 for the values of
    # 'Ecological', pars2 for the values of 'Economic')
    pars1 = 2
    pars2 = 1
    
    # compute the classes for the given list of points (the model returns a
    # python list)
    classes, scores = MyModel.get_class_and_scores_for_weights(arg1_name, arg2_name, [pars1, pars2])
    
    # for using arraydiff, we need however to convert the output to a numpy array
    result = np.array(scores, dtype=float)
    return result


