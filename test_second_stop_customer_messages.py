#!/usr/bin/env python3
"""Focused source-level sanity test for SECOND_STOP_RECOVERY customer messaging."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parent
engine=(ROOT/'monitor'/'engine.py').read_text(encoding='utf-8')
customer=(ROOT/'monitor'/'customer.py').read_text(encoding='utf-8')

ast.parse(engine)
ast.parse(customer)

checks={
    'engine publishes ENTRY_ALERT at first-stop touch':
        "Event.kind=='ENTRY_ALERT'" in engine and "l<=p.target" in engine,
    'pending card uses compact Arabic limit text':
        "نوع الأمر: ليمت شراء — سعر الشراء:" in customer,
    'activation status exists':
        "'تم التفعيل'" in customer,
    'target update status exists':
        "'TARGET':'تحقق الهدف'" in customer,
    'stop update status exists':
        "'STOPPED':'وقف خسارة'" in customer,
    'second-stop messages omit reply markup':
        "reply_markup=None" in customer,
    'ENTRY_ALERT delivery rule exists':
        "payload['_us_kind']=='ENTRY_ALERT'" in customer,
}

failed=[]
for name,ok in checks.items():
    print(('✅' if ok else '❌'),name)
    if not ok:failed.append(name)

print()
print(f'Passed: {len(checks)-len(failed)}/{len(checks)}')
print(f'Failed: {len(failed)}/{len(checks)}')
raise SystemExit(1 if failed else 0)
