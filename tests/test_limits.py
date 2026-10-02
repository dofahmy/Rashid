import unittest,tempfile,os,time,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select,func
from core import database,Lead,Outbox,Setting
from monitor.models import Plan,Stock,Event
from monitor.customer import notice,delivery_allowed,view,Recipient
from monitor.limits import limits,save_limits,bucket,occupied,holding_settings,HOLD_DAYS_KEY,HOLD_PROFIT_KEY
from monitor.engine import advance,DEFAULT_POLICY
from monitor.customer_table import current_rows

class LimitTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.url='sqlite:///'+self.tmp.name+'/limits.db'
  self.env=patch.dict(os.environ,{'DATABASE_URL':self.url,'US_RECOMMENDATIONS_ENABLED':'1','MONITOR_ADMIN_ALERT_IDS':''});self.env.start()
  self.DB=database(self.url);self.ts=int(time.time())//900*900-900
  with self.DB.begin() as s:
   for tid in (10,20):s.add(Lead(telegram_id=tid,market='us',status='trial',completed_at='yes',consent_at='yes',telegram_status='active',step='done'))
 def tearDown(self):self.env.stop();self.tmp.cleanup()
 def plan(self,s,symbol,entry=10):
  s.add(Stock(symbol=symbol,feed_symbol=symbol,market='US',company=symbol,metadata_json='{}',last_price=entry,last_bar=self.ts))
  p=Plan(symbol=symbol,market='US',strategy_version='test',signal_ts=self.ts-1800,last_bar=self.ts,activation_ts=self.ts,state='ACTIVE',entry=entry,paper_entry=entry,stop=entry*.95,target=entry*1.05,atr=entry*.01,score=80,context_json='{}',policy_json=json.dumps(DEFAULT_POLICY))
  s.add(p);s.flush();return p
 def cap(self,s,tid=10,total=1,low=0,mid=1,high=0):save_limits(s,tid,dict(total=total,under_one=low,one_to_100=mid,over_100=high))
 def entry(self,s,p,tid=10):return s.scalar(select(Outbox).where(Outbox.key==f'usrec:{p.id}:ACTIVE:{tid}'))
 def send(self,s,p,tid=10):
  row=self.entry(s,p,tid);row.status='sent';s.flush();return row
 def test_boundaries_and_default_no_unconfigured_broadcast(self):
  self.assertEqual([bucket(v) for v in [.9999,1,99,99.9999,100]],['under_one','one_to_100','one_to_100','one_to_100','over_100'])
  with self.DB.begin() as s:
   self.assertEqual(limits(s,10)['total'],0)
   p=self.plan(s,'ONE');notice(s,p,'ACTIVE',self.ts);s.flush()
   self.assertEqual(s.scalar(select(func.count()).select_from(Recipient)),0)
 def test_total_categories_per_client_and_reopen_slot(self):
  with self.DB.begin() as s:
   self.cap(s,total=3,low=1,mid=1,high=1);self.cap(s,tid=20,total=1,mid=0,high=1)
   plans=[]
   for symbol,price in [('LOW',.9),('LOW2',.8),('MID',99.99),('HIGH',100),('HIGH2',110)]:
    p=self.plan(s,symbol,price);notice(s,p,'ACTIVE',self.ts);plans.append(p)
   s.flush();self.assertEqual(occupied(s,10)['total'],3);self.assertEqual(occupied(s,20)['total'],1)
   self.assertIsNone(self.entry(s,plans[1]));self.assertIsNone(self.entry(s,plans[4]))
   self.send(s,plans[0]);plans[0].state='STOPPED';plans[0].exit_price=.85;notice(s,plans[0],'STOPPED',self.ts+900)
   fresh=self.plan(s,'LOW3',.7);notice(s,fresh,'ACTIVE',self.ts);s.flush()
   self.assertIsNotNone(self.entry(s,fresh));self.assertEqual(occupied(s,10)['total'],3)
 def test_pending_expired_failed_and_uncertain_slots(self):
  with self.DB.begin() as s:
   self.cap(s);p=self.plan(s,'ONE');notice(s,p,'ACTIVE',self.ts);s.flush();row=self.entry(s,p)
   self.assertEqual(occupied(s,10,clock=self.ts+1801)['total'],0)
   row.status='uncertain';self.assertEqual(occupied(s,10,clock=self.ts+1801)['total'],1)
   p.state='DATA_GAP';self.assertEqual(occupied(s,10)['total'],1)
   row.status='failed';self.assertEqual(occupied(s,10)['total'],0)
 def test_lowering_limits_cancels_excess_pending_not_exit(self):
  with self.DB.begin() as s:
   self.cap(s,total=2,mid=2);a=self.plan(s,'A');b=self.plan(s,'B')
   notice(s,a,'ACTIVE',self.ts);notice(s,b,'ACTIVE',self.ts);s.flush();self.cap(s)
   ra=self.entry(s,a);rb=self.entry(s,b)
   self.assertTrue(delivery_allowed(s,ra,json.loads(ra.payload)))
   self.assertFalse(delivery_allowed(s,rb,json.loads(rb.payload)))
   ra.status='sent';self.cap(s,total=0,mid=0);a.state='TARGET';a.exit_price=10.5;notice(s,a,'TARGET',self.ts+900);s.flush()
   out=s.scalar(select(Outbox).where(Outbox.key==f'usrec:{a.id}:TARGET:10'))
   self.assertTrue(delivery_allowed(s,out,json.loads(out.payload)))
 def test_personal_table_results_and_restart(self):
  with self.DB.begin() as s:
   self.cap(s);self.cap(s,tid=20,total=1,mid=0,high=1)
   a=self.plan(s,'PRIVATE_A');b=self.plan(s,'PRIVATE_B',120)
   notice(s,a,'ACTIVE',self.ts);notice(s,b,'ACTIVE',self.ts);self.send(s,a);self.send(s,b,20)
   lead=s.scalar(select(Lead).where(Lead.telegram_id==10))
   self.assertEqual([r['symbol'] for r in current_rows(s,lead)],['PRIVATE_A'])
   self.assertNotIn('PRIVATE_B',view(s,lead,'results')[0])
   b.state='TARGET';b.exit_price=125;self.assertNotIn('PRIVATE_B',view(s,lead,'results')[0])
  with database(self.url)() as s:self.assertEqual(occupied(s,10)['total'],1);self.assertEqual(limits(s,10)['total'],1)
 def test_aged_exit_threshold_notification_idempotence_and_slot(self):
  with self.DB.begin() as s:
   self.cap(s);p=self.plan(s,'AGE');notice(s,p,'ACTIVE',self.ts);self.send(s,p)
   self.assertEqual(holding_settings(s),(3,.5))
   self.assertFalse(advance(s,p,[self.ts+3*86400-900,10,10.2,9.9,10.1,100],1.2))
   self.assertFalse(advance(s,p,[self.ts+3*86400,10,10.2,9.9,10.04,100],1.2))
   end=self.ts+3*86400+900
   self.assertTrue(advance(s,p,[end,10,10.2,9.9,10.05,100],1.2))
   self.assertEqual(p.state,'TIME_EXIT');self.assertEqual(p.exit_price,10.05)
   self.assertFalse(advance(s,p,[end,10,10.2,9.9,10.05,100],1.2))
   s.flush();self.assertEqual(occupied(s,10)['total'],0)
   out=s.scalar(select(Outbox).where(Outbox.key==f'usrec:{p.id}:TIME_EXIT:10'))
   self.assertTrue(delivery_allowed(s,out,json.loads(out.payload)))
   lead=s.scalar(select(Lead).where(Lead.telegram_id==10));self.assertIn('خروج بعد مدة الانتظار',view(s,lead,'results')[0])
   self.assertEqual(s.scalar(select(func.count()).select_from(Event).where(Event.kind=='TIME_EXIT')),1)
 def test_stop_target_and_gap_precede_time_exit(self):
  with self.DB.begin() as s:
   for symbol,h,l,contiguous,expected in [('STOP',11,9,True,'STOPPED'),('TARGET',11,9.9,True,'TARGET'),('GAP',10.2,9.9,False,'DATA_GAP')]:
    p=self.plan(s,symbol);advance(s,p,[self.ts+3*86400,10,h,l,10.1,100],1.2,contiguous)
    self.assertEqual(p.state,expected)
 def test_live_settings_and_weekend_elapsed_calendar_days(self):
  with self.DB.begin() as s:
   p=self.plan(s,'HOLD');s.add(Setting(key=HOLD_DAYS_KEY,value='4'));s.add(Setting(key=HOLD_PROFIT_KEY,value='1'))
   self.assertFalse(advance(s,p,[self.ts+3*86400,10,10.2,9.9,10.1,100],1.2))
   self.assertFalse(advance(s,p,[self.ts+4*86400,10,10.2,9.9,10.09,100],1.2))
   self.assertTrue(advance(s,p,[self.ts+4*86400+900,10,10.2,9.9,10.1,100],1.2))
 def test_admin_validation_and_persistence(self):
  from app import create_app
  from werkzeug.security import generate_password_hash
  app=create_app(self.DB,{'TESTING':True,'SECRET_KEY':'test','SESSION_COOKIE_SECURE':False,'ADMIN_PASSWORD_HASH':generate_password_hash('pw')});c=app.test_client()
  with self.DB() as s:lid=s.scalar(select(Lead.id).where(Lead.telegram_id==10))
  path=f'/leads/{lid}'
  self.assertEqual(c.post(path,data={}).status_code,403)
  with c.session_transaction() as ses:ses['admin']=True;ses['csrf']='test'
  data=dict(csrf='test',status='trial',limit_total='3',limit_under_one='1',limit_one_to_100='1',limit_over_100='1')
  self.assertEqual(c.post(path,data=data).status_code,302)
  for invalid in ['-1','1.5','nan','10001','2']:
   self.assertEqual(c.post(path,data={**data,'limit_total':invalid}).status_code,400)
  self.assertIn('حدود توصيات',c.get(path).get_data(as_text=True))
  settings=dict(csrf='test',minimum_score='60',hold_days='3',hold_min_profit='0.75')
  self.assertEqual(c.post('/stocks/settings',data=settings).status_code,302)
  for changes in [{'hold_days':'0'},{'hold_days':'3.5'},{'hold_min_profit':'nan'},{'hold_min_profit':'-1'}]:
   self.assertEqual(c.post('/stocks/settings',data={**settings,**changes}).status_code,400)
  with database(self.url)() as s:
   self.assertEqual(limits(s,10)['total'],3);self.assertEqual(holding_settings(s),(3,.75))

 def test_views_exclude_pending_uncertain_failed_and_other_clients(self):
  with self.DB.begin() as s:
   self.cap(s,total=5,mid=5)
   lead=s.scalar(select(Lead).where(Lead.telegram_id==10))
   plans=[]
   for name,status in [('SENT_ONLY','sent'),('QUEUED_ONLY','pending'),('UNCERTAIN_ONLY','uncertain'),('FAILED_ONLY','failed')]:
    p=self.plan(s,name);notice(s,p,'ACTIVE',self.ts);s.flush();self.entry(s,p).status=status;plans.append(p)
   self.assertEqual([r['symbol'] for r in current_rows(s,lead)],['SENT_ONLY'])
   for kind in ('current','results'):
    body=view(s,lead,kind)[0]
    self.assertIn('SENT_ONLY',body)
    for name in ('QUEUED_ONLY','UNCERTAIN_ONLY','FAILED_ONLY'):self.assertNotIn(name,body)
   plans[0].state='STOPPED';plans[0].exit_price=9.5
   self.assertEqual(current_rows(s,lead),[])
   self.assertIn('SENT_ONLY',view(s,lead,'results')[0])
 def test_admin_filter_threshold_before_pagination_and_counts(self):
  from app import create_app
  from werkzeug.security import generate_password_hash
  from monitor.customer import MIN_SCORE_KEY
  with self.DB.begin() as s:
   s.add(Setting(key=MIN_SCORE_KEY,value='59.9'))
   for symbol,score in [('BELOW_THRESHOLD',53.5),('EXACT_THRESHOLD',59.9),('ABOVE_THRESHOLD',70)]:
    p=self.plan(s,symbol);p.score=score;p.state='STOPPED';p.exit_price=9.5
  app=create_app(self.DB,{'TESTING':True,'SECRET_KEY':'test','SESSION_COOKIE_SECURE':False,'ADMIN_PASSWORD_HASH':generate_password_hash('pw')});c=app.test_client()
  with c.session_transaction() as ses:ses['admin']=True;ses['csrf']='test'
  body=c.get('/stocks?state=STOPPED').get_data(as_text=True)
  self.assertNotIn('BELOW_THRESHOLD',body)
  self.assertIn('EXACT_THRESHOLD',body);self.assertIn('ABOVE_THRESHOLD',body)
  self.assertIn('2 خطة',body)
  self.assertIn('10.00',body)

 def test_reset_preserves_clients_limits_prices_and_allows_latest_complete_signal(self):
  from monitor.reset import reset_recommendations,RESET_KEY
  from monitor.engine import new_plan
  from monitor.customer import Publication
  from monitor.models import Scan
  from monitor.customer_table import table_entry,TableAccess
  with self.DB.begin() as s:
   self.cap(s);p=self.plan(s,'OLD');notice(s,p,'ACTIVE',self.ts);self.send(s,p)
   lead=s.scalar(select(Lead).where(Lead.telegram_id==10));table_entry(s,lead)
   s.add(Outbox(key='manual:keep',chat_id=10,payload='{}'));s.flush()
   clock=self.ts+1000;reset_recommendations(s,clock)
   for model in (Plan,Event,Recipient,Publication,Scan,TableAccess):
    self.assertEqual(s.scalar(select(func.count()).select_from(model)),0)
   self.assertEqual(s.scalar(select(func.count()).select_from(Lead)),2)
   self.assertEqual(limits(s,10)['total'],1);self.assertIsNotNone(s.get(Stock,'OLD'))
   self.assertEqual(list(s.scalars(select(Outbox.key))),['manual:keep'])
   stock=s.get(Stock,'OLD')
   self.assertEqual(stock.last_bar,0);self.assertEqual(stock.checked_at,'');self.assertEqual(stock.error,'');self.assertEqual(stock.evaluation_json,'{}')
   result=dict(conditional_plan=True,entry_reference=10,stop_reference=9.5,selected_target_price=10.5,atr14=.1,technical_score_100=80)
   self.assertIsNotNone(new_plan(s,stock,result,self.ts,DEFAULT_POLICY))
   # The same signal remains deduplicated even though pre-reset timestamps are allowed.
   self.assertIsNone(new_plan(s,stock,result,self.ts,DEFAULT_POLICY))
 def test_reset_route_csrf_and_running_worker_guard(self):
  from app import create_app
  from werkzeug.security import generate_password_hash
  from monitor.worker import exclusive
  app=create_app(self.DB,{'TESTING':True,'SECRET_KEY':'test','SESSION_COOKIE_SECURE':False,'ADMIN_PASSWORD_HASH':generate_password_hash('pw')});c=app.test_client()
  self.assertEqual(c.post('/stocks/reset').status_code,403)
  with c.session_transaction() as ses:ses['admin']=True;ses['csrf']='test'
  with self.DB.begin() as s:self.plan(s,'KEEP_UNTIL_RESET')
  with exclusive(self.DB,72617368696416) as held:
   self.assertTrue(held);c.post('/stocks/reset',data={'csrf':'test'})
   with self.DB() as s:self.assertEqual(s.scalar(select(func.count()).select_from(Plan)),1)
  with exclusive(self.DB) as held:
   self.assertTrue(held);c.post('/stocks/reset',data={'csrf':'test'})
   with self.DB() as s:self.assertEqual(s.scalar(select(func.count()).select_from(Plan)),1)
  self.assertEqual(c.post('/stocks/reset',data={'csrf':'test'}).status_code,302)
  with self.DB() as s:self.assertEqual(s.scalar(select(func.count()).select_from(Plan)),0)

if __name__=='__main__':unittest.main()
