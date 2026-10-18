.PHONY: help install dev test lint mock doctor bench temp compare study clean

MODEL ?= llama3.2:3b
HOST  ?= http://localhost:11434

help:
	@echo "make install   install LocalMind + deps"
	@echo "make test      run the test suite (no models needed)"
	@echo "make mock      start the fake Ollama daemon on :11500"
	@echo "make doctor    check environment and installed models"
	@echo "make bench     Phase 1 benchmark      (MODEL=$(MODEL))"
	@echo "make temp      Phase 2 determinism    (MODEL=$(MODEL))"
	@echo "make compare   Phase 3 model study"
	@echo "make study     run all three phases end to end"

install:
	pip install -e ".[api,dev]"

dev: install
	pre-commit install || true

test:
	pytest -q

lint:
	ruff check localmind api tools tests

mock:
	python tools/mock_ollama.py --port 11500

doctor:
	localmind doctor --host $(HOST)

bench:
	localmind bench -m $(MODEL) --host $(HOST) --repeats 3

temp:
	localmind temp-study -m $(MODEL) --host $(HOST) --samples 8 --temps 0,0.3,0.7,1.0

compare:
	localmind compare --host $(HOST) --repeats 1

study: bench temp compare
	@echo "All three phases complete. Reports are in ./reports"

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache *.egg-info build dist
