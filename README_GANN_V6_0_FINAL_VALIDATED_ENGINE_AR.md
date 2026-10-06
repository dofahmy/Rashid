# Gann Engine V6.0 — FINAL VALIDATED

هذه النسخة تقفل مرحلة تطوير محرك جان.

## المبادئ النهائية
1. Anchors مصر ثابتة من V4.0 Consensus ولا يعاد تعديلها هنا.
2. كل Forecast تاريخي يبنى As-Of: لا يستخدم Pivot قبل available_date.
3. Backtest مقسوم زمنيًا Train / Late Validation.
4. كل طريقة جان تحصل على Historical Reliability مستقل.
5. Small samples يتم Shrink إلى 50 حتى لا تهيمن ضربة واحدة محظوظة.
6. TOP/LOW لا يظهران كتوقع اتجاهي إلا إذا اجتاز السهم Reliability Gate.
7. إذا فشل الاختبار، يبقى الناتج WATCH_ONLY:
   - نوافذ زمنية للمراقبة
   - دعم/مقاومة رئيسية
   - بدون ادعاء قمة/قاع قادم.
8. Time Window منفصلة عن Direction.
9. Final score ليست Probability وليست توصية تداول.

## Reliability Gate
Direction يحتاج:
- 6 توقعات تاريخية على الأقل.
- Reliability للـLate Validation أو الـFull fallback >= 45.
- Useful rate >= 35%.
- Candidate score >= 55.
- على الأقل طريقة واحدة Validated تاريخيًا.
- المستوى لا يبعد أكثر من 35% عن السعر.

## Method Validation
كل طريقة (Square of Nine / Master 144 / Range Ratios / Mikula...) تقيم على التوقعات السابقة التي شاركت فيها.
Validated method تحتاج:
- 3 عينات على الأقل.
- effective reliability >= 42.

## المخرجات النهائية
- regime = DIRECTIONAL أو WATCH_ONLY.
- دعمين/مقاومتين رئيسيتين فقط.
- حتى 4 Time Watch Windows.
- TOP/LOW فقط عند عبور بوابة التحقق.
- Full + Train + Validation metrics.
- جدول Reliability لكل طريقة.

## بعد Deploy
```bash
python check_gann_engine_final.py COMI.CA
```

ولمراجعة أكثر من سهم:
```bash
python audit_gann_validation.py COMI.CA ADIB.CA QNBA.CA ABUK.CA SWDY.CA
```

المهم ليس أن كل سهم يصبح DIRECTIONAL. النتيجة الصحيحة أحيانًا هي WATCH_ONLY إذا لم يثبت جان تاريخيًا على السهم.
