# Gann V3.9 — Close-based major EGX pivots

التعديلات:
- تحديد القمم/القيعان على المؤشر الصناعي من Close فقط، وليس High/Low الصناعي.
- Breadth / Leaders / Banks أصبحت أدوات تأكيد، وليست مصدر الـpivot.
- حد أدنى زمني 20 جلسة بين LOW وHIGH المقبولين.
- حد الحركة الرئيسية 15% ما زال موجودًا.
- عدم حذف Pivot مقبول في المنتصف بسبب Candidate لاحق فشل.
- تجميع same-type فقط، وعدم دمج LOW مع HIGH القريبين.
- Diagnostic يطبع الـCandidates المرفوضة وأسباب الرفض من 2025 فما بعد.

بعد Deploy:
python check_egx_market_pivots.py

أرسلي:
1) Principal EGX market pivots
2) Rejected/filtered candidates from 2025 onward
