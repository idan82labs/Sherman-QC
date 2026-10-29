"""Independent rational-corridor/error-budget checks, never native inputs."""
import math
from fractions import Fraction
import numpy as np
import pytest
from scripts.extract_step_topology_bend_axes_v6 import certify_axis_corridor, compose_domain_error_budget


def test_positive_rational_corridor_contains_independent_bezier_locus():
    poles=np.array([[0.,0.],[1e-8,1.],[-1e-8,2.],[0.,3.]])
    weights=np.array([1.,4.,.25,2.])
    proof=certify_axis_corridor(poles,weights,1e-7)
    assert proof['status']=='PROVED' and proof['axis_coordinate']==0
    for t in np.linspace(0,1,101):
        b=np.array([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3])*weights
        point=(b[:,None]*poles).sum(axis=0)/b.sum()
        assert abs(point[0]-proof['support_coordinate_metric_mm'])<=proof['coefficient_corridor_mm']
        # Same longitudinal coordinate chooses a point on endpoint chord.
        assert abs(point[0])<=proof['curve_to_endpoint_chord_bound_mm']


@pytest.mark.parametrize('weights',[[1.,0.,1.],[1.,-1.,1.],[1.,float('nan'),1.]])
def test_nonpositive_or_nonfinite_weights_refuse(weights):
    assert certify_axis_corridor([[0,0],[0,1],[0,2]],weights,1e-7)['status']=='REFUSED'


def test_monotone_other_coordinate_required_even_inside_corridor():
    assert certify_axis_corridor([[0,0],[1e-9,2],[0,1]],[1,1,1],1e-7)['status']=='REFUSED'


def test_corridor_budget_is_physical_not_radius_free():
    for radius,expected in [(1.,'PROVED'),(100.,'REFUSED')]:
        assert certify_axis_corridor([[0,0],[radius*1e-8,1],[0,2]],[1,1,1],1e-7)['status']==expected


def test_exact_constant_poles_have_zero_corridor():
    result=certify_axis_corridor([[10000,0],[10000,1],[10000,2]],[1,8,.1],1e-7)
    assert result['status']=='PROVED' and result['coefficient_corridor_mm']==0


def test_individual_terms_cannot_each_spend_source_tolerance():
    assert compose_domain_error_budget(6e-8,2e-8,1e-8,1e-7)['status']=='REFUSED'
    assert compose_domain_error_budget(1e-8,1e-8,1e-8,1e-7)['status']=='PROVED'


def test_outward_sum_bounds_exact_binary_float_terms():
    terms=(1e-8,1e-8,1e-8)
    result=compose_domain_error_budget(*terms,1e-7)
    exact=Fraction.from_float(terms[0])+2*Fraction.from_float(terms[1])+Fraction.from_float(terms[2])
    assert Fraction.from_float(result['total_bound_mm'])>=exact


def test_zero_budget_remains_exact():
    result=compose_domain_error_budget(0.,0.,0.,1e-7)
    assert result['status']=='PROVED' and result['total_bound_mm']==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1e-9,True])
def test_invalid_budget_term_refuses(bad):
    assert compose_domain_error_budget(bad,0.,0.,1e-7)['status']=='REFUSED'


def test_budget_just_above_tolerance_never_rounds_to_acceptance():
    tol=1e-7
    result=compose_domain_error_budget(math.nextafter(tol,math.inf),0.,0.,tol)
    assert result['status']=='REFUSED'
