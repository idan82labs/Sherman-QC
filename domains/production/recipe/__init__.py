"""Pure recipe domain (C01): no file IO, no server imports.

- :mod:`.canonical_sku` - canonical SKU form and collision detection
- :mod:`.sku` - real ZEISS scan-filename parsing (never guesses)
- :mod:`.bend_identity` - ``bend_uid`` and draft-to-draft bend remapping
- :mod:`.builder` - Vendor Setup draft + release request -> PartRecipeV1
"""
