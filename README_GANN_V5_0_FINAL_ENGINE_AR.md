# Gann Engine V5.0 FINAL

هذه هي نسخة المحرك النهائية بعد تثبيت EGX Market Anchors.

## ما الذي يفعله المحرك؟
- مصر: آخر Principal Market LOW + HIGH من Consensus Anchors V4.0، ثم يحول تاريخ السوق إلى extreme فعلي للسهم داخل ±3 جلسات.
- أمريكا: local confirmed pivots بفلتر حركة 4%.
- Price geometry: Square of Nine + Master 144 + Range Ratios.
- Time geometry: Square Low/High/Range + Master 144 + Mikula cell counts.
- يعرض فقط 2 مقاومة/قمة و2 دعم/قاع رئيسيين.
- النوافذ الزمنية القادمة = ±2 يوم حول أقوى cluster، بحد أقصى 270 يوم.
- TOP يفضل time clusters الناتجة من Major LOW، وLOW يفضل clusters الناتجة من Major HIGH. هذا pairing heuristic وليس ادعاء أن الزمن وحده يحدد نوع الانعكاس.
- Backtest تاريخي as-of: الـanchor لا يصبح متاحًا إلا بعد confirmation/available_date، والتوقع القديم يتوقف عندما يصبح الـanchor التالي متاحًا.
- النتيجة التاريخية: تحقق / تحقق جزئي / لم يتحقق.
- Backtest summary: عدد العينات، hit/useful rate، median price/time error.
- Final Decision Score = geometry + anchor stability + shrunk historical reliability. ليست probability.
- واجهة القرار تعرض: وضع السعر داخل النطاق، أقرب دعم، أقرب مقاومة، أقرب نافذة زمنية، ثبات الـAnchor، وموثوقية الاختبار التاريخي.

## إصلاحات مهمة
- EGX يحمل حتى 2200 شمعة للسهم حتى لا تُربط Anchors قديمة بأول شمعة متاحة خطأ.
- أي Market Pivot خارج تاريخ السهم يُهمل بدل nearest-date mapping الخاطئ.
- إصلاح توافق V4.0: market_score يأخذ final_score/quality بدل الاعتماد على مفتاح score القديم.
- مستويات السعر متباعدة Adaptive حسب ATR، فلا تتكدس الخطوط.

## بعد Deploy
1. فحص الـAnchors:
```bash
python final_egx_anchor_decision.py
```
2. فحص المحرك على سهم مصري:
```bash
python check_gann_engine_final.py COMI.CA
```
3. افتحي:
`/gann?market=EGX`

إذا ظهر `FINAL GANN ENGINE CHECK PASSED` فالمحرك جاهز للمرحلة التالية.
