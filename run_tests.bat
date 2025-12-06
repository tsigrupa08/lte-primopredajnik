@echo off
echo Pokretanje unit testova sa coverage...
echo.

REM Provjera da li je pytest-cov instaliran
python -m pip show pytest-cov >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo pytest-cov nije instaliran. Instaliram...
    python -m pip install pytest-cov
)

REM Pokretanje testova i generisanje coverage report-a
pytest --cov=transmitter --cov-report=term-missing --cov-report=html tests

echo.
echo Testovi zavrseni. HTML coverage report se nalazi u folderu "htmlcov".
pause
