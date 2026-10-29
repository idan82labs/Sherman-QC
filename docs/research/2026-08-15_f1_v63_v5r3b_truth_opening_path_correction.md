# F1 V63 V5R3B truth-opening path correction

The first scorer invocation failed before creating its output root and before
reading `private_manifest.json` or any case truth. The supplied V63 argument
was the `primary` directory while the scorer first requires the parent
execution manifest. The sole correction changes the later per-case result path
to `<v63-root>/primary/<case>/semantic_result.json`; no scoring threshold,
prediction, input, truth, rendering semantic, or retention rule changed.

- Failed scorer SHA-256: `da1078ad31c968f274bf7b0066c826e27f95cfdbcad09cde63163bfae597d24a`
- Corrected scorer SHA-256: `c12b96734e026e1b02f4140482588580cbd5117ab860534b921ff0d52e44d0ac`
- Failed invocation truth reads: 0
- Failed invocation output files: 0
- One-time truth opening remains unused and is authorized only with the
  corrected scorer hash above.
