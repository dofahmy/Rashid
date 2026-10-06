# Gann Engine V7.0 — Constance Brown public-method implementation

هذه النسخة تستبدل منطق Gann/Mikula السابق بمنهج مبني على ما هو منشور علنًا عن Constance Brown.

## ما تم تغييره
- حذف Mikula SQ9 وMaster 144 وRange Ratios من المحرك الفعلي.
- Horizontal axis:
  Gann Wheel square-root price objectives من القاع والقمة الرئيسيين.
  الزوايا: 45،90،120،180،240،270،315،360.
  360° = ±2 على sqrt(price)، وبالتالي factor = angle/180.
- Price target لا يعتمد إلا عندما توجد Confluence من أكثر من Anchor/origin.
- Vertical axis:
  Square bar-count time cycles من القمم/القيعان المهمة.
- Disharmonic time noise:
  إذا امتلأت فترة قصيرة بعدد كبير من time clusters فلا تعتبر Confluence جيدة.
- Diagonal axis:
  Brown تنص علنًا أن fixed screen scale مطلوب وأن التحليل على 3 محاور.
  المعادلات الكاملة لـThirty-Second Jewel غير منشورة في المصادر العامة التي استخدمناها،
  لذلك الـbackend يستخدم data-scale fan proxy فقط، ولا يمكنه إنشاء signal وحده.
- Composite Index:
  Momentum(9) of RSI(14) + SMA(3) of RSI(3)
  مع SMA 13 وSMA 33.
  يستخدم لتأكيد اتجاه الانعكاس، وليس لصناعة هدف السعر.
- Directional signal يحتاج:
  1) Horizontal price confluence
  2) Vertical clean time confluence
  3) Diagonal proxy alignment
  4) Composite Index direction confirmation
  5) Walk-forward historical gate
  6) validated Brown method history
- وإلا النتيجة WATCH_ONLY.

## مصادر المنهج العامة
- Constance Brown, Technical Analysis for the Trading Professional, Chapter 9: Gann Analysis.
- Constance Brown, Price and Time, Breakthroughs in Technical Analysis.
- Constance Brown 2023 CMT presentation: fixed screen scale, three axes, square bar count time cycle, Composite Index.
- Optuma Brown/Pythagorean tools documentation.
- StockCharts CMB Composite Index formula.

## نقطة مهمة
هذه ليست نسخة مطابقة حرفيًا للـThirty-Second Jewel proprietary formulas.
المصادر العامة نفسها تقول إن fixed screen scale وPythagorean/three-axis work مهم،
لكنها لا تنشر كل معادلات الكتاب. لذلك أي جزء غير منشور تم وسمه Proxy صراحةً.

## بعد Deploy
```bash
python check_gann_engine_final.py COMI.CA
python audit_gann_validation.py COMI.CA ADIB.CA QNBA.CA ABUK.CA SWDY.CA
```
