# -*- mode: python ; coding: utf-8 -*-
# Single portable exe — run: .\build_windows.ps1  (outputs under ./release/)
# Manual: pyinstaller hamster_scraper.spec --distpath ./release --workpath build/pyinstaller --noconfirm

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

_SELENIUM_SUBMODS = collect_submodules("selenium")
_WDM_SUBMODS = collect_submodules("webdriver_manager")
_WEBVIEW_SUBMODS = collect_submodules("webview")

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static', 'static'),
    ],
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan.on',
        'uvicorn.lifespan.off',
        'multipart',
        'pydantic',
        'starlette.routing',
        'jinja2',
        'yt_dlp',
        'browser_antibot',
        'browser_challenge_wait',
        'challenge_detect',
        'chrome_login_confirm',
        'cookie_health',
        'flaresolverr_client',
        'version',
        'pixiv_ph',
        'pixiv_titles',
        'deep_translator',
        'PIL',
        'PIL.Image',
        'selenium_login_wait',
        'workflow_browser',
        'selenium.webdriver.chrome.webdriver',
        'webview.platforms.winforms',
    ]
    + _SELENIUM_SUBMODS
    + _WDM_SUBMODS
    + _WEBVIEW_SUBMODS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SHUCK3R',
    icon='static/app.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
