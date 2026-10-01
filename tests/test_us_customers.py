import unittest,tempfile,os,json,time,importlib,sys
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select,func
from core import database,Lead,Outbox,handle_update
from monitor.models import Plan,Stock,Event
from monitor.engine import new_plan,DEFAULT_POLICY,advance
from monitor.customer import Publication,Recipient,Preference,notice,view,delivery_allowed,eligible

class CustomerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.url='sqlite:///'+self.tmp.name+'/test.db';self.DB=database(self.url)
  self.ts=int(time.time())//900*900-900
  self.env=patch.dict(os.environ,{'US_RECOMMENDATIONS_ENABLED':'1','MONITOR_ADMIN_ALERT_IDS':'','DATABASE_URL':self.url});self.env.start()
  with self.DB.begin() as s:
   s.add(Stock(symbol='TEST',feed_symbol='TEST',market='US',company='Test',metadata_json='{}',last_price=100.2,last_bar=self.ts))
   for i,market,status in [(1,'us','trial'),(2,'both','subscribed'),(3,'sa','trial'),(4,'us','new'),(5,'us','subscribed')]:
    s.add(Lead(telegram_id=i,market=market,status=status,completed_at='yes',consent_at='yes',telegram_status='active',step='done'))
   s.add(Preference(telegram_id=5,paused=1))
 def tearDown(self):self.env.stop();self.tmp.cleanup()
 def plan(self,s,ts=None):
  ts=self.ts if ts is None else ts
  p=new_plan(s,s.get(Stock,'TEST'),dict(conditional_plan=True,entry_reference=100,stop_reference=98,selected_target_price=104,atr14=2,technical_score_100=70),ts-1800,dict(DEFAULT_POLICY))
  p.state='ACTIVE';p.paper_entry=100.2;p.activation_ts=ts;p.last_bar=ts;s.flush();return p
 def test_only_activated_us_and_both_receive_once(self):
  with self.DB.begin() as s:
   p=self.plan(s);notice(s,p,'ACTIVE',self.ts);notice(s,p,'ACTIVE',self.ts);s.flush()
   self.assertEqual(set(s.scalars(select(Recipient.telegram_id))),{1,2})
   self.assertEqual(s.scalar(select(func.count()).select_from(Outbox)),2)
   self.assertEqual(s.scalar(select(func.count()).select_from(Publication)),1)
 def test_old_activation_and_sa_not_published(self):
  with self.DB.begin() as s:
   p=self.plan(s);notice(s,p,'ACTIVE',self.ts-3600);p.market='SA';notice(s,p,'ACTIVE',self.ts)
   self.assertEqual(s.scalar(select(func.count()).select_from(Publication)),0)
 def test_recheck_entitlement_expiry_and_closed_before_send(self):
  with self.DB.begin() as s:
   p=self.plan(s);notice(s,p,'ACTIVE',self.ts);s.flush();o=s.scalar(select(Outbox));payload=json.loads(o.payload)
   self.assertTrue(delivery_allowed(s,o,payload,clock=self.ts+950))
   self.assertFalse(delivery_allowed(s,o,payload,clock=self.ts+1801))
   l=s.scalar(select(Lead).where(Lead.telegram_id==o.chat_id));l.status='new';self.assertFalse(delivery_allowed(s,o,payload));l.status='trial'
   p.state='STOPPED';self.assertFalse(delivery_allowed(s,o,payload))
 def test_exit_sent_only_after_entry_delivery(self):
  with self.DB.begin() as s:
   p=self.plan(s);notice(s,p,'ACTIVE',self.ts);s.flush();p.state='TARGET';p.exit_price=104
   notice(s,p,'TARGET',self.ts+900);s.flush()
   entry=s.scalar(select(Outbox).where(Outbox.key==f'usrec:{p.id}:ACTIVE:1'))
   exit=s.scalar(select(Outbox).where(Outbox.key==f'usrec:{p.id}:TARGET:1'))
   self.assertFalse(delivery_allowed(s,exit,json.loads(exit.payload)))
   entry.status='sent';self.assertTrue(delivery_allowed(s,exit,json.loads(exit.payload)))
 def test_views_current_month_pnl_and_access(self):
  with self.DB.begin() as s:
   p=self.plan(s);notice(s,p,'ACTIVE',self.ts);s.flush();s.get(Stock,'TEST').last_price=102
   l=s.scalar(select(Lead).where(Lead.telegram_id==1));text,kb=view(s,l,'current')
   self.assertIn('TEST',text);self.assertIn('+1.80%',text);self.assertIn('غير المحقق',text)
   p.state='STOPPED';p.exit_price=98
   self.assertNotIn('TEST',view(s,l,'current')[0]);self.assertIn('TEST',view(s,l,'results')[0]);self.assertIn('-2.20%',view(s,l,'results')[0])
   l.status='new';self.assertNotIn('TEST',view(s,l,'results')[0])
 def test_no_backtest_results_and_no_sa(self):
  with self.DB.begin() as s:
   p=self.plan(s);l=s.scalar(select(Lead).where(Lead.telegram_id==1));self.assertNotIn('TEST',view(s,l,'current')[0])
   p.market='SA';s.add(Publication(plan_id=p.id,published_ts=self.ts,activation_end=self.ts+900));s.flush()
   self.assertNotIn('TEST',view(s,l,'current')[0])
 def test_callback_ack_and_pagination_bounded(self):
  with self.DB.begin() as s:
   p=self.plan(s);notice(s,p,'ACTIVE',self.ts)
   for uid,data in [(101,'us:current:999999999'),(102,'us:results:bad')]:
    handle_update(s,{'update_id':uid,'callback_query':{'id':str(uid),'from':{'id':1},'message':{'chat':{'type':'private','id':1}},'data':data}})
   s.flush();self.assertEqual(s.scalar(select(func.count()).select_from(Outbox).where(Outbox.method=='answerCallbackQuery')),2)
   for o in s.scalars(select(Outbox).where(Outbox.method=='sendMessage')):self.assertLessEqual(len(json.loads(o.payload)['text']),4096)
 def test_month_uses_new_york_and_prior_month_close_included(self):
  clock=datetime(2026,10,1,0,30,tzinfo=ZoneInfo('America/New_York')).timestamp()
  with self.DB.begin() as s:
   p=self.plan(s);p.state='TARGET';p.exit_price=104
   s.add(Publication(plan_id=p.id,published_ts=int(clock-86400),activation_end=int(clock-86400)))
   s.add(Event(key='octclose',plan_id=p.id,symbol='TEST',kind='TARGET',bar_ts=int(clock-900),details_json='{}'));s.flush()
   l=s.scalar(select(Lead).where(Lead.telegram_id==1));self.assertIn('TEST',view(s,l,'results',clock=clock)[0])
 def test_opt_out(self):
  with self.DB.begin() as s:
   l=s.scalar(select(Lead).where(Lead.telegram_id==1))
   handle_update(s,{'update_id':1,'message':{'chat':{'type':'private'},'from':{'id':1},'text':'/stop_us'}});s.flush();self.assertFalse(eligible(s,l))
   handle_update(s,{'update_id':2,'message':{'chat':{'type':'private'},'from':{'id':1},'text':'/resume_us'}});s.flush();self.assertTrue(eligible(s,l))
 def test_restart_does_not_republish(self):
  with self.DB.begin() as s:p=self.plan(s);pid=p.id;notice(s,p,'ACTIVE',self.ts)
  db=database(self.url)
  with db.begin() as s:notice(s,s.get(Plan,pid),'ACTIVE',self.ts);self.assertEqual(s.scalar(select(func.count()).select_from(Outbox)),2)
 def test_gap_after_activation_blocks_new_plan(self):
  with self.DB.begin() as s:
   p=self.plan(s);p.state='DATA_GAP';s.flush()
   self.assertIsNone(new_plan(s,s.get(Stock,'TEST'),{'conditional_plan':True},self.ts+900,DEFAULT_POLICY))
 def test_delivery_strips_private_fields_and_network_ambiguity(self):
  import bot
  with self.DB.begin() as s:p=self.plan(s);notice(s,p,'ACTIVE',self.ts)
  with patch.object(bot,'DB',self.DB),patch.object(bot,'api',return_value={'ok':False,'error_code':0}) as api,patch.object(bot.time,'sleep'):
   bot.deliver();bot.deliver()
   self.assertEqual(api.call_count,2)
   self.assertTrue(all(not k.startswith('_us_') for k in api.call_args.args[1]))
  with self.DB() as s:self.assertEqual(set(s.scalars(select(Outbox.status))),{'uncertain'})

if __name__=='__main__':unittest.main()
