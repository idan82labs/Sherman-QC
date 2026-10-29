"""Supervised job runner (C02): step protocol, subprocess supervisor, job runner.

- :mod:`.worker_protocol` - ``PROGRESS``/``RESULT``/``ERROR`` stdout records
- :mod:`.supervisor` - timeout, RSS ceiling, cancel, process-group kill
- :mod:`.runner` - leases jobs and runs E3 (inspect) + C06 (render), or E1 (compile)
"""
