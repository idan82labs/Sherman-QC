# EXP_CAD_0015 authenticated new physical intake

The browser intake recovered 16 authenticated PLY scans and eight native
SolidWorks CAD files without opening either evaluator-only manifest. All scan
hashes matched the pre-existing SharePoint-link correspondence audits.

Native SolidWorks files are not directly accepted by the production engine.
`cadmpeg 0.5.5` was pinned by release checksum and used to decode them. A new
fail-closed converter admits only tessellation attached to both a decoded body
and a resolved B-rep face. This matters because several files also carry an
unresolved flat/display geometry lane; mixing that lane with the folded body
would corrupt registration.

Seven of eight CAD files produced deterministic engine meshes. `47215000_B`
did not: its Parasolid schema was enumerated but no B-rep topology graph could
be built, so N007/N008 require a source-side STEP export.

The retained registration run used deterministic surface sampling, analytic
CAD-plane extraction, exact point-to-triangle distance, and 3000×1800
oblique/top/side renders. Every rendered artifact was inspected at original
detail. Seven of 14 rendered cases passed the *registration-actionability*
gate; this is not a part-conformance judgment:

- clean actionable: N005, N006, N014;
- actionable with localized/articulated large residuals: N009–N012;
- failed binding or registration actionability: N001–N004, N013, N015, N016;
- blocked CAD decode: N007, N008.

N001/N002 provide strong visual evidence that the delivered `38173001_01`
CAD is not the correct geometry for the `38173000_01` scans. N005/N006 are the
cleanest new registration controls. N009–N012 are the highest-value deviation
cases because stable surrounding CAD/scan structure anchors spatially
localized or articulated disagreement.

Generalization progress indicator:
`GPI-C_SOURCE_INTAKE_AND_ACTIONABILITY_POSITIVE_NOT_F1_PROGRESS`.
Authenticated physical evidence and four actionable CAD families are now
available, but no bend predictions or blinded truth have been scored. The next
bounded experiment is the already selected fixed bend-local direction guard on
the visually actionable cases, followed by candidate-bound visual review and
only then blinded scoring.
