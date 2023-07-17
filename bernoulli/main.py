from asn import average_sample_number
from core import modified_kw
from oc import operating_characteristic
from original_kw import original_kw
from utils import horizon_bound


def calculcate(lam0: float, lam1: float, th0: float, th1: float) -> dict:
    """Calculation of Kiefer-Weiss problem."""
    th, delta = original_kw(lam0, lam1, th0, th1)
    # th = 0.076846178793028
    # delta = -3.78467603923127E-08
    horizon = horizon_bound(lam0, lam1, th0, th1, th)

    cont, accept = modified_kw(horizon, lam0, lam1, th0, th1, th)

    asn = average_sample_number(th, cont)
    asn0 = average_sample_number(th0, cont)
    asn1 = average_sample_number(th1, cont)

    beta = operating_characteristic(cont, accept, th1)
    alpha = 1 - operating_characteristic(cont, accept, th0)

    return {
        'lambda0': lam0,
        'lambda1': lam1,
        'theta0': th0,
        'theta1': th1,
        'theta': th,
        'alpha': alpha,
        'beta': beta,
        'asn': asn,
        'asn0': asn0,
        'asn1': asn1,
        'delta': delta
    }


lam0 = 157.696751972207
lam1 = 193.349705609267
th0 = 0.05
th1 = 0.15

result = calculcate(lam0, lam1, th0, th1)

for key, val in result.items():
    print(f'{key}: {val}')
