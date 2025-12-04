#!/bin/bash

# Ako koristiš virtualno okruženje, otkomentiraj sljedeću liniju
# source .venv/bin/activate

echo "Pokretanje svih testova..."
pytest tests/ --verbose

echo "Pokretanje testova sa coverage..."
pytest --cov=transmitter tests/ --cov-report=term-missing --verbose
