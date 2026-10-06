#!/usr/bin/env python3
from core import database
from monitor.gann_analysis import _load_egx_panel

DB=database()
idx,panel,source=_load_egx_panel(DB)
print("Index source:",source)
print("Index/proxy rows:",len(idx))
print("Stock rows:",len(panel))
if len(idx):
    print("First index/proxy date:",idx.iloc[0]["d"])
    print("Last index/proxy date:",idx.iloc[-1]["d"])
    print("First close:",float(idx.iloc[0]["c"]))
    print("Last close:",float(idx.iloc[-1]["c"]))
