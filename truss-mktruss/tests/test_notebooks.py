import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.parametrize("notebook", ["happy.ipynb"])
def test_notebook_exec(notebook):
    """
    Test for Jupyter notebooks to establish some base exercising.  Future tests can add new notebooks similar to
    happy.ipynb in the test_data directory, then add that file to the above parameterization
    """
    with open(
        f"{os.path.dirname(os.path.realpath(__file__))}/../test_data/{notebook}"
    ) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(nb)
