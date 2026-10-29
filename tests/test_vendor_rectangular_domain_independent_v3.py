"""Independent exact trim-restriction protocol tests, without native geometry."""
import copy
import math
import pytest
from scripts.extract_step_topology_bend_axes_v7 import restrict_bspline_actual_interval


class Curve:
    def __init__(self, defect=None):
        self.knots=[-2.,0.,3.]; self.mult=[3,2,3]
        self.calls=[]; self.defect=defect
    def NbPoles(self): return 5
    def NbKnots(self): return len(self.knots)
    def Knot(self,i): return self.knots[i-1]
    def Multiplicity(self,i): return self.mult[i-1]
    def Degree(self): return 3 if self.calls and self.defect=='degree' else 2
    def IsPeriodic(self): return self.defect=='periodic'
    def FirstParameter(self): return self.knots[0]
    def LastParameter(self): return self.knots[-1]
    def Copy(self): return copy.deepcopy(self)
    def Segment(self,a,b,tolerance):
        self.calls.append((a,b,tolerance))
        if self.defect=='exception': raise RuntimeError('synthetic unsupported restriction')
        self.knots=[a,b]; self.mult=[3,3]
        if self.defect=='snap': self.knots[0]=math.nextafter(a,math.inf)
        if self.defect=='unclamped': self.mult[0]=2
        if self.defect=='discontinuous': self.knots=[a,(a+b)/2,b]; self.mult=[3,3,3]


@pytest.mark.parametrize('interval',[(-2.,3.),(-1.5,-.5),(0.,2.),(math.nextafter(0.,1.),1.)])
def test_exact_interval_identity_and_copy_only_mutation(interval):
    source=Curve(); restricted,proof=restrict_bspline_actual_interval(source,*interval)
    assert proof['status']=='PROVED' and restricted is not source
    assert restricted.calls==[(*interval,0.)]
    assert source.calls==[] and source.knots==[-2.,0.,3.]
    assert proof['actual_trim_interval']==proof['restricted_knot_interval']==list(interval)
    assert proof['interval_clamped'] is False


@pytest.mark.parametrize('interval',[(-2.000000000000001,1.),(0.,math.nextafter(3.,math.inf)),(1.,1.),(2.,1.),(float('nan'),1.),(0.,float('inf'))])
def test_outside_or_invalid_domain_not_clamped_or_segmented(interval):
    source=Curve(); restricted,proof=restrict_bspline_actual_interval(source,*interval)
    assert restricted is None and proof['status']=='REFUSED' and not source.calls


@pytest.mark.parametrize('defect',['periodic','snap','unclamped','degree','discontinuous','exception'])
def test_native_restriction_cannot_silently_change_proof_contract(defect):
    source=Curve(defect); restricted,proof=restrict_bspline_actual_interval(source,.25,1.25)
    assert restricted is None and proof['status']=='REFUSED'
    assert source.calls==[] and source.knots==[-2.,0.,3.]
