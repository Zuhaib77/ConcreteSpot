@echo off
setlocal enabledelayedexpansion

echo ===================================
echo ConcreteSpot Windows Build Script
echo ===================================

cd /d "%~dp0"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt
pip install pyinstaller

echo Building with PyInstaller...
pyinstaller --clean --noconfirm concretespot.spec

echo ===================================
echo Build complete!
echo Output: dist\ConcreteSpot\
echo ===================================

echo.
echo To run:
echo   cd dist\ConcreteSpot ^& ConcreteSpot.exe
echo.
echo To create installer (optional):
echo   Use Inno Setup with the provided .iss file

pause
