# Scan-derived axis and measurement semantics — independent source audit

Date2026-09-09. Read-only axis/radius reviewer findings; no new execution,
physical/Gold access or changes to frozen R0/V1. This is not physical truth.

## Confirmed mechanism

`cad_guided_measurement._measure` fits two scan TLS planes and forms the axis
from their normalized cross product. The signed3D flange sweep is also obtained
from those normals. CAD signs and gates the direction and supplies search masks,
not the measured direction. Circle projection uses this fitted scan axis, not
the CAD axis. Free-circle fits projected scan positions; measured-flange-tangent
also constrains the circle with the scan-fitted planes. Neither copies CAD radius.

Both models treat their estimated planes/axis as fixed during circle fitting.
Thus conditional intrinsic fits do not include joint uncertainty from those
estimates. The fitted `measured_axis_point` uses an along-axis gauge through the
CAD origin; it is a point on the fitted line, not an independently measured3D
position datum. Subtracting it directly from a CAD origin would mix displacement
with arbitrary gauge. Finite endpoints cancel the gauge but are extrema of
scan support already censored by the CAD search windows.

The V1 surface generator is computed from measured radius, sweep, axis and scan
flange normal. It is model-derived, not a directly observed crease. Sampled-not-
physical-trim and unverified-reference labels remain appropriate.

## Axis-error projection mathematics

For radius R, axial coordinate t and projection-axis error theta, an orthonormal
projected coordinate choice gives

`q(t,phi) = (R*cos(theta)*cos(phi) - t*sin(theta), R*sin(phi))`.

At fixed t, the ellipse semiaxes are R*cos(theta) and R. Across footprint length L,
its centers smear by L*sin(theta). Ellipse curvature-radius extrema are
R*cos(theta)^2 and R/cos(theta); at10degrees these are approximately-3.02% and
+1.54% relative to R. These are local curvature limits, not universal fitted-circle
error bounds for arbitrary partial/axially correlated samples. An oblique cylinder
intersection differs: its ellipse semiaxes are R and R/cos(theta).

This is an estimated-scan-axis error risk, not a CAD-axis-copy defect.

## Evidence limits and successor tests

The existing stress generator includes scan-only local tilts of±2degrees before
the common global pose. It scores radius/sweep, not direct axis-direction,
axis-line or finite-carrier error. Common-transform exact tests do not close that
gap. The earlier free-circle stress receipt retains eight radius-limit failures.

Qualify the existing axis estimator using independent direction/line/carrier
truth, scan-only tilts through search boundaries, correlated errors, long axial
footprints and censored support. For deviations, report unoriented axis-angle and
transverse displacement at an explicit datum station, never arbitrary point
subtraction. Jointly propagate registration, flange and axis uncertainty into
radius/sweep/carrier statements; retain weak-support and competing-model
abstentions. Physical trim, acquisition-side and reference compatibility remain
separate from fitted geometry. A new protocol/audit is required before executing
these additional simulations; no current benchmark settings are changed here.
