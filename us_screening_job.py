"""Detached screening worker, one job at a time on the dedicated Railway service."""
import json, os, sys, tempfile, time, traceback
from pathlib import Path

JOBS = Path(os.environ.get('US_SCREENING_JOB_DIR', '/tmp/us_screening_jobs'))

def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.writing-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf8') as fh:
            json.dump(obj, fh, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def main(jobid):
    jobfile = JOBS / (jobid + '.json')
    data = json.loads(jobfile.read_text(encoding='utf8'))
    params = data['params']
    started=time.time()
    def progress(phase, completed, total):
        atomic_json(jobfile, dict(status='running', started_at=started,
             updated_at=time.time(), params=params, phase=phase,
             completed=int(completed), total=int(total)))
    progress('loading', 0, 0)
    try:
        from us_seven_engine import repeat_price_time_screener
        result = repeat_price_time_screener(progress=progress, **params)
        atomic_json(jobfile, dict(status='done', finished_at=time.time(), result=result, params=params))
    except BaseException as err:
        traceback.print_exc()
        atomic_json(jobfile, dict(status='failed', finished_at=time.time(), error=f'{type(err).__name__}: {err}', params=params))
        raise

if __name__ == '__main__':
    if len(sys.argv)!=2 or len(sys.argv[1])!=32: raise SystemExit(2)
    main(sys.argv[1])
