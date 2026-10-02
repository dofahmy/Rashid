# تحديث مصدر الذهب إلى Twelve Data

تم تحويل `XAUUSD` فقط إلى Twelve Data الرسمي على فريم 15 دقيقة.

## Railway Variables
أضف إلى خدمة الـMonitor:

```text
TWELVE_DATA_API_KEY=ضع_المفتاح_هنا
TWELVE_DATA_XAUUSD_SYMBOL=XAU/USD
```

المتغير الثاني اختياري؛ القيمة الافتراضية `XAU/USD`.

## الملفات المعدلة
- `monitor/provider.py`
- `monitor/worker.py`
- `.env.example`

## السجل المتوقع
عند نجاح الجلب:

```text
XA TwelveData reference symbol=XAU/USD bars=... latest=... volume_available=...
XA scan {...}
XA strategy summary {...}
```

لا يتم اختلاق Volume إذا لم يرسله المزود. في هذه الحالة يبقى الحجم = 0 ويظهر مانع الاستراتيجية بوضوح، لأن المطلوب تطبيق نفس قواعد التوصيات الحالية دون تجاوز شرط الحجم.
