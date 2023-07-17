from bernoulli.utils import pmf
import pytest

def test_pmf():
    right_answer = [1.434891e-08, 5.022117e-07, 8.202792e-06, 8.293934e-05,
                    5.805754e-04, 2.980287e-03, 1.159000e-02, 3.477001e-02,
                    8.113003e-02, 1.472360e-01, 2.061304e-01, 2.186231e-01,
                    1.700402e-01, 9.156011e-02, 3.052004e-02, 4.747562e-03]
    to_test = pmf(15, 0.7)

    assert len(to_test) == len(right_answer)
    for i in range(len(to_test)):
        #print(round(to_test[i], 11))
        assert round(to_test[i], 7) == round(right_answer[i], 7)

def test_pmf_2():
    right_answer = [0.512, 0.384, 0.096, 0.008]
    to_test = pmf(3, 0.2)
    assert len(to_test) == len(right_answer)

    for i in range(len(right_answer)):
        assert round(to_test[i], 3) == right_answer[i]

def test_pmf_3():
    right_answer = [0.03125, 0.15625, 0.31250, 0.31250, 0.15625, 0.03125]
    to_test = pmf(5, 0.5)
    assert len(to_test) == len(right_answer)

    for i in range(len(right_answer)):
        assert round(to_test[i], 5) == right_answer[i]




test_pmf()
test_pmf_2()
test_pmf_3()
