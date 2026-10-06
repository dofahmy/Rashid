
#!/usr/bin/env python3
from pathlib import Path
import re, shutil

ROOT=Path(__file__).resolve().parent
candidates=[ROOT/"templates"/"base.html", ROOT/"base.html"]
link='<a href="{{url_for(\'gann.gann_analysis\')}}">تحليل جان</a>'

for p in candidates:
    if not p.exists():
        continue
    txt=p.read_text(encoding="utf-8")
    if "gann.gann_analysis" in txt:
        print(f"{p}: already patched")
        continue
    shutil.copy2(p,p.with_suffix(p.suffix+".before_gann_v3"))
    m=re.search(r'(<a[^>]+url_for\([\'"]stocks[\'"]\)[^>]*>.*?</a>)',txt,flags=re.S)
    if m:
        txt=txt[:m.end()]+"\n"+link+txt[m.end():]
    elif "</nav>" in txt:
        txt=txt.replace("</nav>",link+"\n</nav>",1)
    elif "</header>" in txt:
        txt=txt.replace("</header>",link+"\n</header>",1)
    else:
        txt=link+"\n"+txt
    p.write_text(txt,encoding="utf-8")
    print(f"Patched {p}")

print("Navigation patch complete.")
