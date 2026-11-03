.PHONY: test test-local

test:
	PYTHONPATH=src ./.venv311/bin/python -m pytest tests -q -m "not local"

test-local:
	PYTHONPATH=src ./.venv311/bin/python -m pytest tests -q
