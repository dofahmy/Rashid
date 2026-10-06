# V4.0 — Final EGX Anchor Consensus

هذه النسخة مصممة لتكون نسخة القرار النهائية قبل استكمال منطق جان.

## لماذا تغير المنهج؟
الاختبارات السابقة أثبتت أن Hard Threshold واحد مثل:
- swing >= 15%
- follow-through >= 10%
- score >= 90
قد يرفض Pivot قوي بفارق بسيط جدًا، أو يحذف نقطة سوق مهمة.

## المنهج النهائي
1. Candidate pivot يأتي من Close لمؤشر السوق فقط.
2. يتم قياس:
   - الحركة السابقة
   - Follow-through
   - Breadth لكل السوق
   - القياديات
   - البنوك
3. لا يوجد cliff واحد يقرر القبول.
4. نفس المرشح يُختبر بخمسة نماذج معقولة مختلفة:
   - Balanced
   - Breadth-led
   - Leadership/Banks-led
   - Reversal-led
   - Conservative
5. كل نموذج يختار سلسلة LOW/HIGH متعاقبة باستخدام Dynamic Programming.
6. كل Pivot يحصل على Consensus Votes من 5.
7. الـPrincipal Pivot النهائي يحتاج 3/5 على الأقل.
8. Stability:
   - 100% = اختارته النماذج الخمسة
   - 80% = 4/5
   - 60% = 3/5
9. يوجد فقط Safety Floor صغير:
   - 15 جلسة بين الاتجاهين
   - 6% حركة انتقالية
   الهدف إزالة noise فقط، وليس تعريف الـMajor Pivot.
10. Gann في مصر يستخدم:
    - آخر Principal LOW
    - آخر Principal HIGH
    للتوقعات القادمة.
11. التقييم التاريخي يستخدم كل السلسلة النهائية.

## تعريف الاتجاهات في الأوزان
القاع:
Breadth أعلى وزن لأن المطلوب قاع ظاهر على عدد كبير من الأسهم.

القمة:
القياديات والبنوك وزنها أعلى لأن المطلوب قمة ظاهرة على المؤشر والأسهم الكبيرة، وخاصة البنوك.

## أمر القرار النهائي
بعد Deploy:
```bash
python check_egx_market_pivots.py
```

للقائمة المختصرة فقط:
```bash
python final_egx_anchor_decision.py
```

اعتمدي النسخة إذا كان:
- التسلسل LOW/HIGH متعاقب.
- يوجد Latest LOW و Latest HIGH.
- أغلب الـanchors المهمة Stability >= 60%، ويفضل 80%+.
- تواريخ القمم والقيعان منطقية بصريًا بالنسبة للسوق.
