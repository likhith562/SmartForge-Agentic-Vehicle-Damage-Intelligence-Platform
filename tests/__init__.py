"""
SmartForge test suite.

Run all tests:
    pytest tests/ -v

Run a single module:
    pytest tests/test_fraud_layer.py -v

Run with coverage:
    pytest tests/ --cov=src --cov-report=term-missing -v

Environment
-----------
Tests are designed to run without GPU, without real API keys, and without
a live MongoDB connection.  Every external dependency is either mocked or
bypassed via the same BYPASS_FRAUD / cfg flag pattern used in production.

Markers
-------
    unit        — pure function tests, no I/O
    integration — tests that touch the filesystem or the SQLite fallback
    slow        — tests that load heavy model mocks (> 1 s)

Add to pytest.ini or pyproject.toml to register markers:
    [pytest]
    markers =
        unit: pure function tests
        integration: touches filesystem or SQLite
        slow: heavyweight setup
"""
