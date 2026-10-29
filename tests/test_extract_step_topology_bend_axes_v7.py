"""Pure exact active-interval seam tests; no CAD imports or physical inputs."""
import copy
import pytest
from scripts.extract_step_topology_bend_axes_v7 import restrict_bspline_actual_interval,certify_axis_corridor


class Spline:
    def __init__(self,periodic=False,snap=False):
        self.knots=[0.,1.,2.]; self.mult=[3,2,3]; self.periodic=periodic
        self.snap=snap; self.segment_calls=[]
    def IsPeriodic(self): return self.periodic
    def Knot(self,i): return self.knots[i-1]
    def NbKnots(self): return len(self.knots)
    def NbPoles(self): return 5
    def Degree(self): return 2
    def Multiplicity(self,i): return self.mult[i-1]
    def FirstParameter(self): return self.knots[0]
    def LastParameter(self): return self.knots[-1]
    def Copy(self): return copy.deepcopy(self)
    def Segment(self,a,b,tol):
        self.segment_calls.append((a,b,tol))
        self.knots=[a+1e-12 if self.snap else a,b]; self.mult=[3,3]


def test_copy_exact_interval_without_mutating_source():
    original=Spline(); restricted,p=restrict_bspline_actual_interval(original,.2,.8)
    assert p['status']=='PROVED' and restricted is not original
    assert restricted.segment_calls==[(.2,.8,0.)]
    assert original.knots==[0.,1.,2.] and not original.segment_calls
    assert p['actual_trim_interval']==p['restricted_knot_interval']==[.2,.8]


@pytest.mark.parametrize('a,b',[(-1.,1.),(1.,3.),(2.,1.),(0.,float('inf'))])
def test_invalid_or_outside_interval_never_segments(a,b):
    original=Spline(); restricted,p=restrict_bspline_actual_interval(original,a,b)
    assert restricted is None and p['status']=='REFUSED'
    assert not original.segment_calls


def test_periodic_refused_before_copy_segment():
    restricted,p=restrict_bspline_actual_interval(Spline(periodic=True),0.,1.)
    assert restricted is None and p['reason']=='PCURVE_PERIODIC_ACTIVE_INTERVAL_UNSUPPORTED'


def test_native_snap_is_not_actual_trim_identity():
    restricted,p=restrict_bspline_actual_interval(Spline(snap=True),.2,.8)
    assert restricted is None
    assert p['reason']=='PCURVE_ACTIVE_INTERVAL_RESTRICTION_CHANGED_BOUNDS'


def test_active_straight_coefficients_pass_while_whole_carrier_fails():
    whole=[[0.,20.],[1.,-10.],[2.,0.],[3.,0.],[4.,0.]]
    assert certify_axis_corridor(whole,[1.]*5,1e-7)['status']=='REFUSED'
    assert certify_axis_corridor(whole[2:],[1.]*3,1e-7)['status']=='PROVED'


def test_curved_active_coefficients_stay_negative():
    assert certify_axis_corridor([[0.,0.],[1.,1.],[2.,0.]],[1.,1.,1.],1e-7)['status']=='REFUSED'


def test_coefficient_budget_precedes_native_mutation():
    class Huge(Spline):
        def NbPoles(self): return 10000
        def Copy(self): raise AssertionError('must not copy oversized carrier')
    restricted,p=restrict_bspline_actual_interval(Huge(),0.,1.)
    assert restricted is None and p['reason']=='PCURVE_COEFFICIENT_BUDGET'


def test_nonclamped_result_is_not_accepted():
    class Nonclamped(Spline):
        def Segment(self,a,b,tol):
            super().Segment(a,b,tol); self.mult=[2,3]
    restricted,p=restrict_bspline_actual_interval(Nonclamped(),.2,.8)
    assert restricted is None and p['reason']=='PCURVE_ACTIVE_INTERVAL_RESTRICTION_CHANGED_BOUNDS'


def test_native_exception_is_typed_refusal():
    class Unsupported(Spline):
        def Segment(self,a,b,tol): raise RuntimeError('unsupported')
    restricted,p=restrict_bspline_actual_interval(Unsupported(),.2,.8)
    assert restricted is None and p['reason']=='PCURVE_ACTIVE_INTERVAL_RESTRICTION_UNSUPPORTED'
    assert p['exception_type']=='RuntimeError'


def test_outer_knots_do_not_authorize_extrapolation_outside_actual_domain():
    class Unclamped(Spline):
        def FirstParameter(self): return .5
        def LastParameter(self): return 1.5
        def Copy(self): raise AssertionError('outside actual domain must refuse before copy')
    restricted,p=restrict_bspline_actual_interval(Unclamped(),.2,.8)
    assert restricted is None and p['reason']=='PCURVE_TRIM_OUTSIDE_UNDERLYING_PARAMETER_DOMAIN'
    assert p['underlying_knot_interval']==[0.,2.]
    assert p['underlying_actual_parameter_interval']==[.5,1.5]
