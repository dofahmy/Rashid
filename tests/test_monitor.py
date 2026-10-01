import unittest, tempfile, json, os, sys, asyncio
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select,func
from core import database, Outbox
from monitor.models import Stock, Plan, Event, Candle, Scan
from monitor.engine import new_plan,advance,DEFAULT_POLICY
from monitor.worker import initialize,exclusive,expected_slot,due,apply_stock,scan_market
from monitor.strategy import clean,evaluate,hourly
from app import create_app
from werkzeug.security import generate_password_hash
FIXTURE=Path(__file__).parent/'fixtures/COHU.json'
class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.DB=database('sqlite:///'+self.temp.name+'/monitor.db')
        self.env=patch.dict(os.environ,{'MONITOR_ADMIN_ALERT_IDS':'','ADMIN_TELEGRAM_IDS':''});self.env.start()
        with self.DB.begin() as s:s.add(Stock(symbol='TEST',feed_symbol='TEST',market='US',company='Test',metadata_json='{}',sharia_label='غير متوافق'))
    def tearDown(self):self.env.stop();self.temp.cleanup()
    def make_plan(self,s,ts=1000):
        r=dict(conditional_plan=True,entry_reference=100.,stop_reference=98.,selected_target_price=104.,atr14=2.,technical_score_100=70.)
        return new_plan(s,s.get(Stock,'TEST'),r,ts,dict(DEFAULT_POLICY))
    def test_new_plan_dedup_and_no_sharia_filter(self):
        with self.DB.begin() as s:
            self.assertIsNotNone(self.make_plan(s));self.assertIsNone(self.make_plan(s))
            self.assertEqual(s.scalar(select(func.count()).select_from(Plan)),1)
            self.assertEqual(s.scalar(select(func.count()).select_from(Event)),1)
    def test_activation_requires_volume_then_later_retest(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);advance(s,p,[1900,100,101,99.5,100.5,1],1.09);self.assertEqual(p.state,'WAITING')
            advance(s,p,[2800,100.2,101,100.1,100.5,1],1.2);self.assertEqual(p.state,'RETEST');self.assertIsNone(p.paper_entry)
            advance(s,p,[3700,100.4,100.7,100.1,100.2,1],.9);self.assertEqual(p.state,'ACTIVE');self.assertEqual(p.paper_entry,100.2)
            n=s.scalar(select(func.count()).select_from(Event));self.assertFalse(advance(s,p,[3700,100.4,100.7,100.1,100.2,1],.9))
            self.assertEqual(s.scalar(select(func.count()).select_from(Event)),n)
    def test_wick_does_not_trigger(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);advance(s,p,[1900,99,101,99,99.8,1],2);self.assertEqual(p.state,'WAITING')
    def test_conservative_stop_first_and_gap_exit(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);p.state='ACTIVE';p.paper_entry=100;advance(s,p,[1900,97,105,96,102,1],1)
            self.assertEqual(p.state,'STOPPED');self.assertEqual(p.exit_price,97)
            s.flush();e=s.scalar(select(Event).where(Event.kind=='STOPPED'));self.assertTrue(json.loads(e.details_json)['both_target_and_stop'])
    def test_target_and_missing_data_review(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);p.state='ACTIVE';advance(s,p,[1900,101,104.2,100,103,1],1);self.assertEqual(p.state,'TARGET')
            p=self.make_plan(s,2000);advance(s,p,[3800,100,101,99.8,100.5,1],2,False);self.assertEqual(p.state,'DATA_GAP')
    def test_retest_and_waiting_expiry(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);p.state='RETEST'
            for i in range(3):advance(s,p,[1900+i*900,101,102,100.8,101,1],1)
            self.assertEqual(p.state,'EXPIRED')
            p=self.make_plan(s,5000);p.waiting_bars=19;advance(s,p,[5900,99.5,99.8,99,99.5,1],.8);self.assertEqual(p.state,'EXPIRED')
    def test_fill_rr_and_target_before_entry(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);p.state='RETEST';advance(s,p,[1900,100,103,100.1,103,1],1);self.assertEqual(p.state,'CANCELLED')
            p=self.make_plan(s,2000);advance(s,p,[2900,100,104.1,99.5,103,1],2);self.assertEqual(p.state,'MISSED')
    def test_admin_outbox_dedup(self):
        with patch.dict(os.environ,{'MONITOR_ADMIN_ALERT_IDS':'123'}):
            with self.DB.begin() as s:self.make_plan(s);self.make_plan(s)
        with self.DB() as s:
            self.assertEqual(s.scalar(select(func.count()).select_from(Outbox)),1);self.assertEqual(s.scalar(select(Outbox)).chat_id,123)
    def test_restart_preserves_pinned_levels(self):
        with self.DB.begin() as s:
            p=self.make_plan(s);pid=p.id;advance(s,p,[1900,100,101,100,100.5,1],2)
        db=database('sqlite:///'+self.temp.name+'/monitor.db')
        with db.begin() as s:
            p=s.get(Plan,pid);self.assertEqual(p.state,'RETEST');self.assertEqual(p.entry,100)
            advance(s,p,[2800,100.1,100.6,100,100.2,1],1);self.assertEqual(p.state,'ACTIVE')
    def test_closed_bars_hour_alignment_dst_and_sunday(self):
        raw=json.loads(FIXTURE.read_text());b=clean(raw,'US');last=b[-1][0]
        self.assertEqual(clean(raw,'US',last+899)[-1][0],last-900);self.assertEqual(clean(raw,'US',last+900)[-1][0],last)
        self.assertTrue(all(datetime.fromtimestamp(a[0],ZoneInfo('America/New_York')).minute==30 for a in hourly(b,'US')))
        for day,hour in [('2026-09-30',13),('2026-12-01',14)]:
            clock=datetime.fromisoformat(day+'T09:45:45').replace(tzinfo=ZoneInfo('America/New_York')).timestamp()
            self.assertEqual(datetime.fromtimestamp(expected_slot('US',clock),ZoneInfo('UTC')).hour,hour)
        sunday=datetime(2026,10,4,11,tzinfo=ZoneInfo('Asia/Riyadh')).timestamp();self.assertTrue(due('SA',sunday));self.assertFalse(due('US',sunday))
    def test_snapshot_regression_and_worker_idempotency(self):
        raw=json.loads(FIXTURE.read_text());b=clean(raw,'US');clock=b[-1][0]+945
        meta=dict(symbol='COHU',feed_symbol='COHU',market_key='US',name_ar='',name='Cohu',sharia_label='غير متوافق',sharia_code='3',reported_price='68.17')
        dates=sorted({datetime.fromtimestamp(a[0],ZoneInfo('America/New_York')).date() for a in b})
        r=evaluate(meta,dates,raw,clock);self.assertTrue(r['conditional_plan']);self.assertEqual(r['entry_reference'],68.5)
        self.assertEqual(r['stop_reference'],67.14);self.assertEqual(r['selected_target_price'],70.56);self.assertEqual(r['technical_score_100'],67.2)
        with self.DB.begin() as s:s.add(Stock(symbol='COHU',feed_symbol='COHU',market='US',company='Cohu',metadata_json=json.dumps(meta)))
        self.assertEqual(apply_stock(self.DB,'COHU',raw,b[-1][0],b,dict(DEFAULT_POLICY),clock),(1,0,True))
        self.assertEqual(apply_stock(self.DB,'COHU',raw,b[-1][0],b,dict(DEFAULT_POLICY),clock),(0,0,True))
        with self.DB() as s:self.assertEqual(s.scalar(select(Plan)).state,'WAITING');self.assertEqual(s.scalar(select(func.count()).select_from(Candle)),len(b))
    def test_local_scanner_lock(self):
        with exclusive(self.DB) as first:
            self.assertTrue(first)
            with exclusive(self.DB) as second:self.assertFalse(second)
    def test_frozen_universe_idempotent_initialization(self):
        db=database('sqlite:///'+self.temp.name+'/universe.db');self.assertEqual(initialize(db),{'US':5691});initialize(db)
        with db() as s:self.assertEqual(s.scalar(select(func.count()).select_from(Stock)),5691)
    def test_dashboard_access_control(self):
        app=create_app(self.DB,{'TESTING':True,'SECRET_KEY':'test','SESSION_COOKIE_SECURE':False,'ADMIN_PASSWORD_HASH':generate_password_hash('password')});client=app.test_client()
        self.assertEqual(client.get('/stocks').status_code,302);self.assertEqual(client.get('/stocks/feed').status_code,302)
        with client.session_transaction() as s:s['admin']=True
        self.assertEqual(client.get('/stocks').status_code,200);self.assertEqual(client.get('/stocks/feed').status_code,200)
        with self.DB.begin() as s:p=self.make_plan(s);pid=p.id
        self.assertEqual(client.get('/stocks/'+str(pid)).status_code,200)
    def test_invalid_feed_does_not_commit(self):
        raw=json.loads(FIXTURE.read_text());raw['chart']['result'][0]['meta']['currency']='EUR'
        with self.DB.begin() as s:s.add(Stock(symbol='COHU',feed_symbol='COHU',market='US',company='Cohu',metadata_json='{}'))
        with self.assertRaises(Exception):apply_stock(self.DB,'COHU',raw,1,[],dict(DEFAULT_POLICY),2)
        with self.DB() as s:self.assertEqual(s.get(Stock,'COHU').last_bar,0)
    def sample_stock(self,s):
        meta=dict(symbol='COHU',feed_symbol='COHU',market_key='US',name_ar='',name='Cohu',sharia_label='غير متوافق',sharia_code='3',reported_price='68.17')
        s.add(Stock(symbol='COHU',feed_symbol='COHU',market='US',company='Cohu',metadata_json=json.dumps(meta)))
    def test_partial_feed_does_not_create_stale_plan(self):
        raw=json.loads(FIXTURE.read_text());b=clean(raw,'US');expected=b[-1][0];clock=expected+945
        with self.DB.begin() as s:self.sample_stock(s)
        short=json.loads(json.dumps(raw));d=short['chart']['result'][0]
        n=sum(t<expected for t in d['timestamp']);d['timestamp']=d['timestamp'][:n]
        for key in ('open','high','low','close','volume'):d['indicators']['quote'][0][key]=d['indicators']['quote'][0][key][:n]
        result=apply_stock(self.DB,'COHU',short,expected,b,dict(DEFAULT_POLICY),clock)
        self.assertEqual(result,(0,0,False))
        with self.DB() as s:
            self.assertEqual(s.scalar(select(func.count()).select_from(Plan)),0)
            self.assertEqual(s.get(Stock,'COHU').error,'awaiting_expected_closed_bar')
        self.assertEqual(apply_stock(self.DB,'COHU',raw,expected,b,dict(DEFAULT_POLICY),clock),(1,0,True))
    def test_atomic_rollback_bars_events_watermark(self):
        raw=json.loads(FIXTURE.read_text());b=clean(raw,'US')
        with self.DB.begin() as s:self.sample_stock(s)
        with patch('monitor.worker.new_plan',side_effect=RuntimeError('simulated crash')):
            with self.assertRaises(RuntimeError):apply_stock(self.DB,'COHU',raw,b[-1][0],b,dict(DEFAULT_POLICY),b[-1][0]+945)
        with self.DB() as s:
            self.assertEqual(s.get(Stock,'COHU').last_bar,0)
            self.assertEqual(s.scalar(select(func.count()).select_from(Candle)),0)
            self.assertEqual(s.scalar(select(func.count()).select_from(Event)),0)
    def test_scan_integration_and_retry_watermark(self):
        raw=json.loads(FIXTURE.read_text());b=clean(raw,'US');clock=b[-1][0]+945
        reference=json.loads(json.dumps(raw));reference['chart']['result'][0]['meta']['symbol']='AAPL'
        with self.DB.begin() as s:
            s.delete(s.get(Stock,'TEST'));self.sample_stock(s)
        class Fake:
            async def fetch(self,symbol,bootstrap=False):return reference if symbol=='AAPL' else raw
        asyncio.run(scan_market(self.DB,Fake(),'US',clock,dict(DEFAULT_POLICY)))
        asyncio.run(scan_market(self.DB,Fake(),'US',clock,dict(DEFAULT_POLICY)))
        with self.DB() as s:
            scans=s.scalars(select(Scan).order_by(Scan.id)).all()
            self.assertEqual(scans[0].new_plans,1);self.assertEqual(scans[0].ok,1)
            self.assertEqual(scans[1].total,0);self.assertEqual(scans[1].new_plans,0)
            self.assertEqual(s.scalar(select(func.count()).select_from(Plan)),1)
    def test_reference_delay_retains_stock_watermark(self):
        raw=json.loads(FIXTURE.read_text());b=clean(raw,'US');clock=b[-1][0]+86400+945
        raw['chart']['result'][0]['meta']['symbol']='AAPL'
        class Fake:
            async def fetch(self,*args):return raw
        asyncio.run(scan_market(self.DB,Fake(),'US',clock,dict(DEFAULT_POLICY)))
        with self.DB() as s:
            self.assertEqual(s.scalar(select(Scan)).status,'waiting_feed')
            self.assertEqual(s.get(Stock,'TEST').last_bar,0)
    def test_railway_requires_shared_postgres(self):
        with patch.dict(os.environ,{'RAILWAY_PROJECT_ID':'testing','DATABASE_URL':''}):
            with self.assertRaisesRegex(RuntimeError,'DATABASE_URL'):database()
            with self.assertRaisesRegex(RuntimeError,'PostgreSQL'):database('sqlite:///'+self.temp.name+'/bad.db')
if __name__=='__main__':unittest.main()
