"""Scan ingest (C02): ports, PLY/STL probe, SMB-safe polling hotfolder.

- :mod:`.ports` - resolver / recipe-provider ports (frozen dataclasses, Protocols)
- :mod:`.ply_probe` - structural probe of PLY and STL files
- :mod:`.hotfolder` - the polling inbox watcher
- :mod:`.sku_resolver` - ``ScanRecipeResolver`` over C01 ``parse_scan_filename``
  (imports C01 code, so it is not imported here)
"""
