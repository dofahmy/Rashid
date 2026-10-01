"""Score the historical 40 conditional plans; not a probability model."""
import json, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'selected_40_plans.json'
def score(r):
    # Trend 25: short EMA alignment 10, full 15m trend 8, hourly alignment 7.
    trend=10*(r['ema20']>=r['ema50'])+8*bool(r['trend_up'])+7*(r['hour_trend_up'] is True)
    # Momentum 20: MACD above signal 10; balanced RSI gets at most 10.
    rsi=r['rsi14']
    rsi_points=10 if 45<=rsi<=65 else max(0,10*(rsi-35)/10) if rsi<45 else max(0,10*(75-rsi)/10)
    momentum=10*(r['macd']>=r['macd_signal'])+rsi_points
    # Same time of day volume only, linearly reaches full weight at 2x.
    volume=15*max(0,min(1,(r['same_time_volume_ratio']-0.8)/1.2))
    # Room relative to the common 3% objective; unknown resistance is not unlimited room.
    resistance=r.get('next_local_resistance')
    room_ratio=max(0,(resistance-r['entry_reference'])/(r['entry_reference']*.03)) if resistance is not None else None
    room=20*min(1,room_ratio) if room_ratio is not None else 10
    # Use the SAME 3% objective for every stock, avoiding reward for stretching target.
    rr=r['reward_risk_3pct']
    risk_reward=15*max(0,min(1,(rr-1)/2))
    # Planned trigger proximity in ATR, not activation confirmation.
    distance=max(0,(r['entry_reference']-r['feed_last_price'])/r['atr14'])
    entry=5*max(0,1-min(1,distance/2))
    parts=dict(trend=trend,momentum=momentum,relative_volume=volume,resistance_room=room,risk_reward=risk_reward,entry_proximity=entry)
    return dict(r,score_components={k:round(v,2) for k,v in parts.items()},technical_score_100=round(sum(parts.values()),1),resistance_room_ratio_to_3pct=room_ratio,score_type='experimental_historical_technical_priority_not_success_probability',client_ready=False)
