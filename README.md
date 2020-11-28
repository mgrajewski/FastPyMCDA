# FastPyMCDA
python implementation of MCDA models in python optimised for speed

FastPyMCDA is a python implementation of outperforming models for Multiple-criteria Decision Analysis (MCDA). Currently, two models are implemented:
- SAW
- Promethee II with SAW aggregation

FastPymCDA runs with python 3 and requires pandas and numpy.

For the sake of computational speed, we herein heavily rely on NumPy. This makes FastPyMCDA particularly suitable when numerous decisions have to be made, e.g. in the
context of parameter studies or a stochastic analysis of a decision model.
The program is divided in a wrapper class (wrapper.py) and the actual models (model.py), which are both located in the directory src.
The workflow is as follows:
1) The model parameters (weights and characteristics) are stored in an Excel file. We provide a sample file ('model_parameters_test.xlsx') in the directory test which
   shows the expected formatting of the data.
2) Open a python shell.
3) import the wrapper class, e.g. by 'import wrapper as wr', if the current working directory is FastPyMCDA/test
4) create a wrapper object by e.g. 'mywrapper = wr.wrapper(my_Excel_file.xlxs, my_worksheet_in_this_file). You need to provide the Excel file from which the model
   parameters are to be read and the worksheet within this file.
5) mywrapper.get_class_for_weights('category1', 'category2', [list_of_values_cat1, list_of_values_cat2) returns the outcome of the decision model if the
   weights of the categories category1 and category2 (these names must be provided in the row 'categories' in the Excel file) are set to the values in the lists,
   respectively. The list of indices of the alternatives is 0-based; the indices are in the order provided in the Excel-file.

For further documentation, we refer to the source code and to FastPyMCDA – a Python Package for Multiple-criteria Decision Analysis, submitted to the
Journal of Open Research Software.

We provide a number of tests. They require the python packages pytest and pytest-arraydiff to be installed. The tests can be started in a shell
(not a python interactive shell) in the directory 'test'. The tests are run by  'py.test --arraydiff'.
