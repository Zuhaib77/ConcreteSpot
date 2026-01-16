#!/bin/bash
set -e

echo "==================================="
echo "ConcreteSpot Linux Build Script"
echo "==================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller

echo "Building with PyInstaller..."
pyinstaller --clean --noconfirm concretespot.spec

echo "==================================="
echo "Build complete!"
echo "Output: dist/ConcreteSpot/"
echo "==================================="

echo ""
echo "To run:"
echo "  cd dist/ConcreteSpot && ./ConcreteSpot"
echo ""
echo "To create AppImage (optional):"
echo "  See: https://appimage.org/"
