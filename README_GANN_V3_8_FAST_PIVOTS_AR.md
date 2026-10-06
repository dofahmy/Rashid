# V3.8 — Fast EGX Major Market Pivots

إصلاح أداء فقط، بدون تخفيف شروط اختيار القاع/القمة الرئيسية.

المشكلة في V3.7:
كل Candidate Pivot كان يعيد فلترة DataFrame الكامل (~385k صف)
لكل واحد من 210 سهم، ثم القياديات والبنوك. مع التاريخ الكامل أصبح بطيئًا جدًا.

الإصلاح:
- تجهيز بيانات كل سهم مرة واحدة.
- تخزين dates/highs/lows في arrays.
- استخدام binary search للوصول للجلسة الأقرب.
- حساب breadth/leaders/banks من arrays بدل DataFrame filtering المتكرر.
- نفس شروط V3.7 تبقى كما هي:
  score >= 90
  follow-through >= 10%
  major swing >= 15%
  low breadth >= 55%
  high breadth >= 30%

بعد Deploy:
python check_egx_market_pivots.py

السكريبت يعرض الآن Calculation time.
