"""Explicit development reset; caller must hold both worker locks."""
import time
from sqlalchemy import delete,or_,update
from core import Outbox,Setting
from .models import Stock,Plan,Event,Scan
from .customer import Publication,Recipient
from .customer_table import TableAccess

RESET_KEY='recommendations_reset_at'

def reset_recommendations(s,clock=None):
    clock=int(time.time() if clock is None else clock)
    # These namespaces contain stock alerts only; registration/CRM messages stay.
    s.execute(delete(Outbox).where(or_(Outbox.key.like('usrec:%'),Outbox.key.like('market:%'))))
    for model in (Recipient,Publication,Event,Plan,Scan,TableAccess):
        s.execute(delete(model))

    # Development reset must make every stock eligible for a fresh evaluation.
    # Keep the universe, cached last price and candle history, but clear the
    # processing watermark/status so the next monitor scan re-checks all stocks.
    s.execute(update(Stock).values(
        last_bar=0,
        checked_at='',
        error='',
        evaluation_json='{}',
    ))

    setting=s.get(Setting,RESET_KEY)
    if setting is None:setting=Setting(key=RESET_KEY);s.add(setting)
    setting.value=str(clock)
    return clock
