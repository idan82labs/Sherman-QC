"""Pure rational coefficient and total-error budgets; no native input."""
import math
import numpy as np
import pytest
from scripts.extract_step_topology_bend_axes_v6 import (
    certify_axis_corridor,compose_domain_error_budget,certify_rectangle_segments)


def test_positive_rational_coefficients_bound_small_nonconstant_axis_locus():
    proof=certify_axis_corridor([[0.,0.],[1.,2e-8],[3.,0.]],[1.,2.,1.],1e-7)
    assert proof['status']=='PROVED' and proof['axis_coordinate']==1
    assert proof['coefficient_corridor_mm']==pytest.approx(1e-8)
    assert proof['curve_to_endpoint_chord_bound_mm']==pytest.approx(2e-8)
    assert proof['coefficient_ranges_metric_mm']==[[0.,3.],[0.,2e-8]]
    assert type(proof['coefficient_corridor_mm']) is float


def test_beyond_fixed_fraction_is_not_accepted_by_samples_or_rounding():
    proof=certify_axis_corridor([[0.,0.],[1.,1e-6],[3.,0.]],[1.,2.,1.],1e-7)
    assert proof['status']=='REFUSED'
    assert proof['reason']=='PCURVE_OUTSIDE_CORRIDOR_OR_BACKTRACK'


@pytest.mark.parametrize('poles,weights',[
    ([[0.,0.],[3.,0.],[1.,0.]],[1.,2.,1.]),
    ([[0.,0.],[1.,0.],[3.,0.]],[1.,0.,1.]),
    ([[0.,0.],[1.,0.],[3.,0.]],[1.,-1.,1.]),
    ([[0.,0.],[math.nan,0.],[3.,0.]],[1.,2.,1.]),
    ([[0.,0.],[0.,0.],[0.,0.]],[1.,2.,1.])])
def test_backtrack_nonpositive_weights_nonfinite_and_degenerate_refuse(poles,weights):
    assert certify_axis_corridor(poles,weights,1e-7)['status']=='REFUSED'


def test_total_additivity_prevents_spending_native_tolerance_three_times():
    proof=compose_domain_error_budget(4e-8,2e-8,3e-8,1e-7)
    assert proof['status']=='REFUSED'
    assert proof['total_bound_mm']==pytest.approx(1.1e-7)
    assert compose_domain_error_budget(2e-8,1e-8,1e-8,1e-7)['status']=='PROVED'


@pytest.mark.parametrize('value',[math.inf,math.nan,-1.,True])
def test_invalid_error_terms_refuse(value):
    assert compose_domain_error_budget(value,0.,0.,1e-7)['status']=='REFUSED'


def test_live_metric_proof_residual_is_python_float_without_json_roundtrip():
    p=[[0.,0.],[3.,0.],[3.+5e-8,40.],[0.,40.]]
    segments=[{'start':row,'end':p[(i+1)%4],'edge_id':i+1} for i,row in enumerate(p)]
    proof=certify_rectangle_segments(segments,1e-7)
    assert proof['status']=='PROVED' and type(proof['maximum_residual_mm']) is float
