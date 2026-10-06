# Gann Backend V3

هذه النسخة لا تحتاج تعديل `app.py`.

## الملفات التي تستبدل/تضاف
- `wsgi.py` — يسجل Blueprint جان مباشرة.
- `monitor/gann_web.py` — routes `/gann` و`/gann/<symbol>`.
- `monitor/gann_analysis.py` — محرك التحليل.
- `templates/gann_analysis.html`
- `templates/gann_symbol.html`
- `patch_gann_nav.py` — يضيف تاب تحليل جان للـbase.html الحالي بدون استبدال بقية التصميم.

## التركيب الدائم في GitHub
انسخي الملفات بنفس المسارات إلى جذر repo، ثم شغلي مرة واحدة محليًا قبل الـcommit:
```bash
python patch_gann_nav.py
```

ثم ارفعي الملفات كلها إلى GitHub واعملي Redeploy لخدمة Web.

## لماذا V3؟
- لا يعتمد على حقن routes داخل `app.py`.
- يسجل جان من `wsgi.py` مباشرة، لذلك `/gann` يظهر عند بدء Gunicorn.
- لا يستخدم Plotly/CDN لأن CSP الحالي يمنع السكربتات الخارجية؛ الشارت SVG من السيرفر.
- يعرض القمتين والقاعين القادمين وآخر قمتين وقاعين متوقعين سابقين.
