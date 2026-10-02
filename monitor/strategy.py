import json,math,statistics,pathlib,collections,datetime,importlib.util
from zoneinfo import ZoneInfo
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prior_score',((ROOT/'rank_stocks.py') if (ROOT/'rank_stocks.py').exists() else ROOT.parent.parent/'output/rajih_rank_100/rank_stocks.py'));MODEL=importlib.util.module_from_spec(spec);spec.loader.exec_module(MODEL)
CONFIG={'SA':{'tz':'Asia/Riyadh','start':600,'end':900,'currency':'SAR','ref':'2222.SR'},'US':{'tz':'America/New_York','start':570,'end':960,'currency':'USD','ref':'AAPL'},'XA':{'tz':'America/New_York','start':0,'end':1440,'currency':'USD','ref':'XAUUSD'}}
def local(t,m):return datetime.datetime.fromtimestamp(t,ZoneInfo(CONFIG[m]['tz']))
def clean(raw,m,asof=None):
 d=raw['chart']['result'][0];q=d['indicators']['quote'][0];b=[];cfg=CONFIG[m]
 for i,t in enumerate(d.get('timestamp',[])):
  v=[q[k][i] for k in ['open','high','low','close','volume']];dt=local(t,m);minute=dt.hour*60+dt.minute
  if (asof is not None and t+900>asof) or dt.minute%15 or dt.second or not cfg['start']<=minute<cfg['end']:continue
  if any(x is None or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v) or min(v[:4])<=0 or v[4]<0:continue
  if not (v[2]<=v[0]<=v[1] and v[2]<=v[3]<=v[1]):continue
  b.append([t,*v])
 return sorted({a[0]:a for a in b}.values())
def hourly(b,m):
 cfg=CONFIG[m];groups={};tz=ZoneInfo(cfg['tz'])
 for a in b:
  dt=local(a[0],m);minute=dt.hour*60+dt.minute;start=cfg['start']+((minute-cfg['start'])//60)*60
  if start+60>cfg['end']:continue
  key=datetime.datetime(dt.year,dt.month,dt.day,start//60,start%60,tzinfo=tz).timestamp();groups.setdefault(key,[]).append(a)
 return [[t,g[0][1],max(x[2] for x in g),min(x[3] for x in g),g[-1][4],sum(x[5] for x in g)] for t,g in sorted(groups.items()) if len(g)==4 and {x[0] for x in g}=={t+900*i for i in range(4)}]
def ema(c,n):
 v=c[0];out=[]
 for x in c:v+=(x-v)*2/(n+1);out.append(v)
 return out
def indicators(b):
 c=[a[4] for a in b];h=[a[2] for a in b];l=[a[3] for a in b];sma={f'sma{n}':statistics.mean(c[-n:]) for n in [20,50,100,200]};tr=[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(c))];chg=[c[i]-c[i-1] for i in range(1,len(c))];atr=statistics.mean(tr[:14]);gain=statistics.mean(max(x,0) for x in chg[:14]);loss=statistics.mean(max(-x,0) for x in chg[:14])
 for t,g in zip(tr[14:],chg[14:]):atr=(atr*13+t)/14;gain=(gain*13+max(g,0))/14;loss=(loss*13+max(-g,0))/14
 macd=[a-b for a,b in zip(ema(c,12),ema(c,26))];return dict(sma,ema20=ema(c,20)[-1],ema50=ema(c,50)[-1],rsi14=100-100/(1+gain/loss) if loss else 100 if gain else 50,atr14=atr,macd=macd[-1],macd_signal=ema(macd,9)[-1],trend_up=c[-1]>sma['sma20']>sma['sma50'] and c[-1]>sma['sma200'])
def pivots(b,col):
 out=[]
 for i in range(2,len(b)-2):
  left=[a[col] for a in b[i-2:i]];right=[a[col] for a in b[i+1:i+3]];x=b[i][col]
  if col==3 and x<=min(left+right) and x<max(left) and x<max(right):out.append(i)
  if col==2 and x>=max(left+right) and x>min(left) and x>min(right):out.append(i)
 return out
def tick(price,m):
 if m in ('US','XA'):return .0001 if price<1 else .01
 return .01 if price<25 else .02 if price<50 else .05 if price<100 else .1 if price<250 else .2 if price<500 else .5
def rounded(price,m,up):
 t=tick(price,m)
 for _ in range(3):
  v=(math.ceil(price/t-1e-9) if up else math.floor(price/t+1e-9))*t;nt=tick(v,m)
  if nt==t:return round(v,6),t
  t=nt
 return round(v,6),t
def evaluate(r,calendar,raw,asof,reference_slots=None):
 m=r['market_key'];cfg=CONFIG[m];sym=r['feed_symbol'];
 out={'symbol':r['symbol'],'feed_symbol':sym,'market':m,'company':r['name_ar'] or r['name'],'currency':cfg['currency'],'sharia_label':r['sharia_label'],'sharia_code':r['sharia_code'],'sharia_filter_applied':False,'timeframe':'15m','eligible':False,'conditional_plan':False,'client_ready':False,'blockers':[]}
 try:
  d=raw['chart']['result'][0];meta=d['meta'];b=clean(raw,m,asof);h=hourly(b,m)
 except Exception as e:out['blockers']=['parse_'+type(e).__name__];return out
 out.update(bars=len(b),hourly_bars=len(h),source_url=raw.get('_retrieval',{}).get('url','https://query1.finance.yahoo.com/v8/finance/chart/'+sym+'?range=60d&interval=15m&includePrePost=false'),retrieved_utc=raw.get('_retrieval',{}).get('retrieved_utc'),data_granularity=meta.get('dataGranularity'))
 if meta.get('dataGranularity')!='15m':out['blockers'].append('wrong_data_granularity')
 expected_type='CURRENCY' if m=='XA' else 'EQUITY'
 if meta.get('instrumentType')!=expected_type:out['blockers'].append('feed_not_'+expected_type.lower())
 if meta.get('currency')!=cfg['currency']:out['blockers'].append('currency_mismatch')
 if meta.get('symbol')!=sym:out['blockers'].append('feed_symbol_mismatch')
 if len(b)<200:out['blockers'].append('fewer_than_200_complete_bars')
 if out['blockers']:return out
 out['eligible']=True;out.update(indicators(b));out.update(bar_start_local=local(b[-1][0],m).isoformat(),bar_end_local=local(b[-1][0]+900,m).isoformat(),signal_bar_close=b[-1][4],feed_last_price=b[-1][4],feed_reported_price=meta.get('regularMarketPrice'),sahm_reported_price=r['reported_price'],last_volume=b[-1][5])
 grid=[datetime.datetime(day.year,day.month,day.day,minute//60,minute%60,tzinfo=ZoneInfo(cfg['tz'])).timestamp() for day in calendar for minute in range(cfg['start'],cfg['end'],15) if day.isoformat()<=local(b[-1][0],m).date().isoformat()];expected=set([t for t in (reference_slots if reference_slots is not None else grid) if t<=b[-1][0]][-21:]);out['missing_last21']=len(expected-{x[0] for x in b})
 if out['missing_last21']:out['blockers'].append('missing_last21_expected_market_slots')
 prior_dates=set([day for day in calendar if day<local(b[-1][0],m).date()][-20:]);prior=[x for x in b[:-1] if local(x[0],m).date() in prior_dates and local(x[0],m).strftime('%H:%M')==local(b[-1][0],m).strftime('%H:%M')];avg=statistics.mean(x[5] for x in prior) if prior else 0;out.update(same_time_sessions=len(prior),same_time_average_volume=avg,same_time_volume_ratio=b[-1][5]/avg if avg>0 else 0)
 if len(prior)!=20 or avg<=0:out['blockers'].append('missing_20_same_time_session_baseline')
 if b[-1][5]<=0:out['blockers'].append('no_volume_in_signal_bar')
 out['volume_confirmed_now']=out['same_time_volume_ratio']>=1.1
 out['hour_trend_up']=indicators(h)['trend_up'] if len(h)>=200 and local(h[-1][0],m).date().isoformat()==local(b[-1][0],m).date().isoformat() else None
 out['hour_bar_end_local']=local(h[-1][0]+3600,m).isoformat() if h else None
 if not 35<=out['rsi14']<=75:out['blockers'].append('RSI_outside_35_75')
 if not(out['ema20']>=out['ema50'] or out['macd']>=out['macd_signal']):out['blockers'].append('no_short_trend_or_momentum_improvement')
 if out['atr14']<=0:out['blockers'].append('zero_ATR')
 if out['blockers']:return out
 entry,tk=rounded(b[-1][2]+.05*out['atr14'],m,True);out.update(entry_reference=entry,planning_tick=tk)
 supports=[i for i in pivots(b,3) if i>=len(b)-80 and b[i][3]<min(b[-1][4],entry)]
 if not supports:out['blockers'].append('no_recent_confirmed_support');return out
 support=b[supports[-1]][3];stop,_=rounded(support-.25*out['atr14'],m,False);risk=entry-stop;out.update(support_reference=support,support_time_local=local(b[supports[-1]][0],m).isoformat(),stop_reference=stop)
 if stop<=0 or risk<=0:out['blockers'].append('invalid_stop_or_risk');return out
 out['risk_pct']=100*risk/entry
 for n in [3,4,5]:
  # Executable objectives rounded up; actual percentage may exceed requested by one tick.
  target,_=rounded(entry*(1+n/100),m,True);out[f'target_{n}pct']=target;out[f'reward_risk_{n}pct']=(target-entry)/risk
 available=[n for n in [3,4,5] if out[f'reward_risk_{n}pct']>=1.5]
 if not available:out['blockers'].append('RR_below_1_5_at_all_requested_targets');return out
 n=available[0];out.update(selected_target_pct=n,selected_target_price=out[f'target_{n}pct'],selected_reward_risk=out[f'reward_risk_{n}pct'],actual_selected_target_pct=100*(out[f'target_{n}pct']/entry-1))
 resistance=[b[i][2] for i in pivots(b,2) if b[i][2]>entry+.1*out['atr14']];high20=max(x[2] for x in b[-21:-1]);out['prior_high20']=high20
 if high20>entry+.1*out['atr14']:resistance.append(high20)
 out['next_local_resistance']=min(resistance,default=None);out['resistance_before_selected_target']=out['next_local_resistance'] is not None and out['next_local_resistance']<out['selected_target_price'];out['conditional_plan']=True;out['hour_confirmation_required_for_priority']=True;out['priority_candidate']=out['hour_trend_up'] is True and out['volume_confirmed_now'] and not out['resistance_before_selected_target'];out['unknown_resistance']=out['next_local_resistance'] is None;out['setup_type']='اتجاه قصير' if out['ema20']>=out['ema50'] else 'ارتداد مبكر';out['plan_status']='خطة مشروطة؛ انتظار إغلاق تفعيل وحجم وإعادة اختبار';out['activation_status']='pending_future_close_above_entry_volume_110pct_and_retest';out['targets_type']='percentage_objectives_not_forecasts';out['news_status']='not_reviewed';out['execution_costs_status']='not_included';out['score_type']='experimental_technical_priority_not_probability';return MODEL.score(out)
