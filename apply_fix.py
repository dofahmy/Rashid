from pathlib import Path
import ast
import sys

path = Path(sys.argv[1]) if len(sys.argv)>1 else Path('monitor/us_tp_daily.py')
src = path.read_text(encoding='utf-8')
old_query = "done=set(s.scalars(text('SELECT symbol FROM us_tp_daily_done WHERE day=:day AND ok=TRUE'),{'day':day}).all())"
new_query = "done=set(s.scalars(text('SELECT symbol FROM us_tp_daily_done WHERE day=:day'),{'day':day}).all())"
old_error = """            errors=[(sym,e) for sym,e in results if e]
            attempts+=len(results)"""
new_error = """            errors=[(sym,e) for sym,e in results if e]
            # Record failed attempts for this trading day. Never mark them as successful.
            # Retried on a new trading day; successful rows are never overwritten.
            if errors:
                with DB.begin() as s:
                    s.execute(text('''INSERT INTO us_tp_daily_done(day,symbol,ok)
                        VALUES(:day,:symbol,FALSE)
                        ON CONFLICT(day,symbol) DO NOTHING'''),
                        [{'day': day, 'symbol': sym} for sym, _ in errors])
            attempts+=len(results)"""
for original, replacement, label in [(old_query,new_query,'done query'),(old_error,new_error,'error handling')]:
    count=src.count(original)
    if count != 1:
        raise SystemExit(f'ABORT: expected exactly 1 match for {label}, got {count}. File unchanged.')
    src=src.replace(original,replacement,1)
ast.parse(src)
backup = path.with_name(path.name+'.before_skip_fix')
if backup.exists():
    raise SystemExit(f'ABORT: backup already exists: {backup}')
backup.write_text(path.read_text(encoding='utf-8'),encoding='utf-8')
path.write_text(src,encoding='utf-8')
print('PATCHED:',path,'BACKUP:',backup)
print('NOTE: scanned_count counts successful symbols ONLY. Failures remain ok=FALSE and publication remains disabled until all 5691 succeed.')
