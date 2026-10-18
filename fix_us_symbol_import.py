"""One-time safe compatibility fix for US Screening: restores is_us_symbol.
Run from /app in the USA Railway service. Does not alter Repeat calculations.
"""
from pathlib import Path
import ast
import shutil

path = Path('us_seven_engine.py')
if not path.is_file():
    raise SystemExit('ERROR: Run this in the folder containing us_seven_engine.py (/app).')
source = path.read_text(encoding='utf-8')
tree = ast.parse(source)
existing = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
if 'is_us_symbol' in existing:
    print('is_us_symbol already exists — no changes made.')
else:
    patch = '''\n\ndef is_us_symbol(value):\n    """Accept US common tickers/share classes; reject Egyptian/Saudi suffixes."""\n    import re\n    ticker = str(value or '').strip().upper()\n    if ticker.endswith(('.CA', '.SR')):\n        return False\n    return re.fullmatch(r'[A-Z]{1,5}(?:[.-][A-Z])?', ticker) is not None\n'''
    ast.parse(source + patch)
    backup = path.with_suffix('.py.before_us_symbol_fix.bak')
    shutil.copy2(path, backup)
    path.write_text(source.rstrip() + '\n' + patch, encoding='utf-8')
    print('Patched:', path, '| Backup:', backup)

from us_seven_engine import is_us_symbol
for ticker, expected in [('AAPL',True), ('BRK.B',True), ('ATLC.CA',False), ('2222.SR',False)]:
    assert is_us_symbol(ticker) == expected, ticker
print('PASS: US symbol function imports and market exclusions work.')
