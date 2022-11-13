from preprocessing.sanity_checks import check_all 
import pytest 

@pytest.mark.order(1)
def test_check_all(datadir):
    check_all(datadir)