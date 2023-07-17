from bernoulli.core import modified_kw
import json

PATH_TO_PARAMS = 'params_for_tests.json'

with open(PATH_TO_PARAMS) as f:
    data = json.load(f)


cont_true_value = [[1.0, 0.0, None, None, None, None],
 [1.0, 0.0, 0.0, None, None, None],
 [1.0, 0.0, 0.0, 0.0, None, None],
 [1.0, 0.0, 0.0, 0.0, 0.0, None],
 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

accept_true_value = [[0.0, 0.0, None, None, None,None],
 [1.0, 0.0, 0.0, None, None, None],
 [1.0, 0.0, 0.0, 0.0, None, None],
 [1.0, 0.0, 0.0, 0.0, 0.0, None],
 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

def test_modified_kw_1():
    data_for_test = data[0]
    cont = modified_kw(5, data_for_test["lamda0"], data_for_test["lambda1"], data_for_test["th0"], data_for_test["th1"], data_for_test["th"])[0]
    accept = modified_kw(5, data_for_test["lamda0"], data_for_test["lambda1"], data_for_test["th0"], data_for_test["th1"], data_for_test["th"])[1]

    assert len(accept) == len(accept_true_value)

    #actually testing
    for i, j in range(len(accept), len(accept)):
        assert accept_true_value[i][j] == accept[i][j]

    for i, j in range(len(cont_true_value), len(cont_true_value)):
        assert cont_true_value[i][j] == cont[i][j]


test_modified_kw_1()