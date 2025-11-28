@echo off
REM LLM Finance Framework - Windows Development Helper
echo LLM Finance Framework - Development Tools
echo.

if "%1"=="help" goto help
if "%1"=="setup" goto setup
if "%1"=="test" goto test
if "%1"=="lint" goto lint
if "%1"=="check" goto check
if "%1"=="version" goto version
if "%1"=="clean" goto clean
if "%1"=="collab" goto collab
goto help

:help
echo Available commands:
echo   help      - Show this help message
echo   setup     - Install development dependencies
echo   test      - Run test suite
echo   lint      - Run code quality checks
echo   check     - Run full development check suite
echo   version   - Version management (use: dev version [get^|bump^|tag])
echo   clean     - Clean build artifacts
echo   collab    - Collaboration tools (use: dev collab [setup^|sync^|branch^|pr^|status])
echo.
echo Examples:
echo   dev setup
echo   dev check
echo   dev version get
echo   dev version bump patch
goto end

:setup
echo Setting up development environment...
python -m pip install -e .[dev]
goto end

:test
echo Running tests...
python -m pytest tests/ -v --cov=src/ --cov-report=term-missing
goto end

:lint
echo Running code quality checks...
flake8 src/ tests/ --max-line-length=127 --extend-ignore=E203,W503
black --check --diff src/ tests/
isort --check-only --diff src/ tests/
goto end

:check
echo Running full development check suite...
call :lint
python -m pytest tests/ -v --cov=src/
goto end

:version
if "%2"=="" (
    echo Usage: dev version [get^|set^|bump^|tag] [args...]
    goto end
)
python scripts/version.py %2 %3 %4 %5
goto end

:clean
echo Cleaning build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
if exist .coverage del .coverage
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .mypy_cache rmdir /s /q .mypy_cache
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
for /r %%f in (*.pyc) do del "%%f"
echo Clean complete.
goto end

:collab
if "%2"=="" (
    echo Usage: dev collab [setup^|sync^|branch^|pr^|status]
    goto end
)
python scripts/collaborate.py %2 %3 %4 %5
goto end

:end
