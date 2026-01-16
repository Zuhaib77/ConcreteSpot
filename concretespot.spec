# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

src_path = Path('src')
models_path = Path('models')
assets_path = Path('assets')

datas = [
    (str(models_path), 'models'),
    (str(assets_path), 'assets'),
]

hiddenimports = [
    'ultralytics',
    'ultralytics.nn',
    'ultralytics.nn.tasks',
    'ultralytics.utils',
    'ultralytics.utils.callbacks',
    'ultralytics.engine',
    'ultralytics.engine.model',
    'ultralytics.engine.predictor',
    'ultralytics.engine.results',
    'ultralytics.data',
    'ultralytics.models',
    'ultralytics.models.yolo',
    'ultralytics.models.yolo.detect',
    'torch',
    'torch.nn',
    'torch.cuda',
    'torchvision',
    'torchvision.models',
    'cv2',
    'PIL',
    'PIL.Image',
    'numpy',
    'reportlab',
    'reportlab.lib',
    'reportlab.lib.pagesizes',
    'reportlab.platypus',
    'sqlite3',
    'PySide6',
    'PySide6.QtWidgets',
    'PySide6.QtCore',
    'PySide6.QtGui',
]

a = Analysis(
    ['src/main.py'],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib.backends',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ConcreteSpot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ConcreteSpot',
)
