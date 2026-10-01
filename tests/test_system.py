import unittest, tempfile, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sqlalchemy import select,func
from werkzeug.security import generate_password_hash
from core import database,Lead,Outbox,Activity,handle_update,normalize_phone
from app import create_app

class SystemTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.DB=database('sqlite:///'+self.temp.name+'/test.db'); self.i=0
        self.app=create_app(self.DB,{'TESTING':True,'SECRET_KEY':'test-secret','SESSION_COOKIE_SECURE':False,'ADMIN_PASSWORD_HASH':generate_password_hash('test-password-123')})
        self.client=self.app.test_client()
    def tearDown(self): self.temp.cleanup()
    def update(self,text=None,callback=None,contact=None,user=123):
        self.i+=1; who={'id':user,'first_name':'Test','username':'testuser'}; msg={'chat':{'id':user,'type':'private'},'from':who}
        if text is not None: msg['text']=text
        if contact: msg['contact']=contact
        u={'update_id':self.i}
        if callback: u['callback_query']={'id':str(self.i),'from':who,'message':msg,'data':callback}
        else: u['message']=msg
        with self.DB.begin() as s: handle_update(s,u)
        return u
    def register(self,user=123):
        self.update('/start campaign_A',user=user); self.update(callback='market:both',user=user); self.update('دعاء',user=user); self.update('+201012345678',user=user); self.update(callback='confirm',user=user)
    def login(self):
        self.client.get('/login')
        with self.client.session_transaction() as s: csrf=s['csrf']
        return self.client.post('/login',data={'csrf':csrf,'password':'test-password-123'})
    def csrf(self):
        with self.client.session_transaction() as s: return s['csrf']
    def test_full_registration_idempotent_and_restart(self):
        self.update('/start campaign_A'); self.update(callback='market:both'); self.update('دعاء')
        # Reopen a session to simulate restart: the phone step is in the database.
        with self.DB() as s: self.assertEqual(s.scalar(select(Lead)).step,'phone')
        self.update('+٢٠١٠١٢٣٤٥٦٧٨'); last=self.update(callback='confirm')
        with self.DB.begin() as s: handle_update(s,last)
        with self.DB() as s:
            l=s.scalar(select(Lead)); self.assertEqual((l.name,l.phone,l.market,l.source),('دعاء','+201012345678','both','campaign_A')); self.assertTrue(l.consent_at); self.assertEqual(s.scalar(select(func.count()).select_from(Activity)),1)
        self.update('/start')
        with self.DB() as s: self.assertEqual(s.scalar(select(func.count()).select_from(Lead)),1)
    def test_invalid_phone_contact_and_stale_button(self):
        self.update('/start'); self.update(callback='confirm'); self.update(callback='market:sa'); self.update('Name'); self.update('01012345678'); self.update(contact={'user_id':999,'phone_number':'201012345678'})
        with self.DB() as s: self.assertEqual(s.scalar(select(Lead)).step,'phone')
        self.update(contact={'user_id':123,'phone_number':'201012345678'}); self.update(callback='confirm')
        with self.DB() as s: self.assertTrue(s.scalar(select(Lead)).completed_at)
    def test_cancel_edit_preserves_confirmed_profile(self):
        self.register(); self.update(callback='edit'); self.update(callback='market:us'); self.update('Changed'); self.update('+966512345678'); self.update('/cancel')
        with self.DB() as s:
            l=s.scalar(select(Lead)); self.assertEqual(l.name,'دعاء'); self.assertEqual(l.market,'both'); self.assertEqual(l.phone,'+201012345678')
    def test_auth_csrf_dashboard_export_and_followup(self):
        self.register(); self.assertEqual(self.client.get('/').status_code,302); self.assertEqual(self.client.get('/export.csv').status_code,302); self.assertEqual(self.client.post('/login',data={'password':'test-password-123'}).status_code,403)
        self.assertEqual(self.login().status_code,302)
        r=self.client.get('/'); self.assertEqual(r.status_code,200); self.assertIn('دعاء',r.text)
        r=self.client.post('/leads/1',data={'csrf':self.csrf(),'status':'trial','owner':'خدمة العملاء','follow_up':'2026-10-01','note':'تم الاتصال'})
        self.assertEqual(r.status_code,302)
        self.assertEqual(self.client.get('/leads/1').status_code,200)
        self.assertEqual(self.client.get('/?market=both&status=trial&q=دعاء').status_code,200)
        r=self.client.get('/export.csv?complete=yes'); self.assertEqual(r.status_code,200); self.assertIn("'+201012345678",r.text)
        self.assertEqual(self.client.post('/leads/1/message',data={'csrf':self.csrf(),'message':'أهلًا بك'}).status_code,302)
        with self.DB() as s: self.assertEqual(s.scalar(select(Lead)).status,'trial')
    def test_no_message_without_consent(self):
        self.update('/start'); self.login(); self.client.post('/leads/1/message',data={'csrf':self.csrf(),'message':'Test'})
        with self.DB() as s: self.assertEqual(s.scalar(select(func.count()).select_from(Activity)),0)
    def test_phone(self):
        self.assertEqual(normalize_phone('00966 512-345-678'),'+966512345678'); self.assertIsNone(normalize_phone('abc+201012345678')); self.assertIsNone(normalize_phone('01012345678'))
    def test_start_photo_caption_buttons_and_dedup(self):
        import json
        from core import BRAND
        update=self.update('/start')
        with self.DB.begin() as s: handle_update(s,update)
        with self.DB() as s:
            rows=s.scalars(select(Outbox)).all()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0].method,'sendPhoto')
            payload=json.loads(rows[0].payload)
            self.assertIn(BRAND,payload['caption'])
            self.assertLessEqual(len(payload['caption']),1024)
            self.assertTrue(payload['_welcome_photo'])
            self.assertEqual(len(payload['reply_markup']['inline_keyboard']),3)
    def test_html_escaped_and_csv_formula_safe(self):
        self.register(); self.login()
        with self.DB.begin() as s: s.get(Lead,1).name='<script>alert(1)</script>'
        self.assertNotIn('<script>alert(1)</script>',self.client.get('/').text)
        with self.DB.begin() as s: s.get(Lead,1).name='=HYPERLINK("bad")'
        self.assertIn("'=HYPERLINK",self.client.get('/export.csv').text)
if __name__=='__main__': unittest.main()

class DeliveryTests(unittest.TestCase):
    def setUp(self):
        import importlib
        self.temp=tempfile.TemporaryDirectory(); os.environ['DATABASE_URL']='sqlite:///'+self.temp.name+'/worker.db'
        self.bot=importlib.import_module('bot'); self.bot.DB=database(os.environ['DATABASE_URL'])
        with self.bot.DB.begin() as s:
            s.add(Lead(telegram_id=999,name='Client'))
            s.add(Outbox(key='first',chat_id=999,payload='{"chat_id":999,"text":"first"}'))
            s.add(Outbox(key='second',chat_id=999,payload='{"chat_id":999,"text":"second"}'))
    def tearDown(self): self.temp.cleanup()
    def test_photo_multipart_and_missing_asset_fallback(self):
        from unittest.mock import patch,Mock
        import json
        reply=Mock(); reply.json.return_value={'ok':True}
        payload={'chat_id':999,'caption':'Welcome','reply_markup':{'inline_keyboard':[]},'_welcome_photo':True}
        with patch.object(self.bot.requests,'post',return_value=reply) as post:
            self.assertTrue(self.bot.api('sendPhoto',payload)['ok'])
            self.assertIn('files',post.call_args.kwargs)
            self.assertEqual(json.loads(post.call_args.kwargs['data']['reply_markup']),{'inline_keyboard':[]})
            self.assertNotIn('_welcome_photo',post.call_args.kwargs['data'])
        with patch.object(self.bot.Path,'is_file',return_value=False),patch.object(self.bot.requests,'post',return_value=reply) as post:
            self.bot.api('sendPhoto',payload)
            self.assertTrue(post.call_args.args[0].endswith('/sendMessage'))
            self.assertEqual(post.call_args.kwargs['json']['text'],'Welcome')
    def test_retry_order_and_blocked_state(self):
        from unittest.mock import patch
        with patch.object(self.bot,'api',return_value={'ok':False,'error_code':429,'parameters':{'retry_after':10}}) as api,patch.object(self.bot.time,'sleep'):
            self.bot.deliver(); self.assertEqual(api.call_count,1)
        with self.bot.DB() as s:
            self.assertEqual(s.get(Outbox,1).status,'pending'); self.assertEqual(s.get(Outbox,2).attempts,0)
        with self.bot.DB.begin() as s: s.get(Outbox,1).next_at=0
        with patch.object(self.bot,'api',side_effect=[{'ok':True},{'ok':False,'error_code':403}]),patch.object(self.bot.time,'sleep'):
            self.bot.deliver()
        with self.bot.DB() as s:
            self.assertEqual(s.get(Outbox,1).status,'sent'); self.assertEqual(s.get(Outbox,2).status,'failed'); self.assertEqual(s.scalar(select(Lead)).telegram_status,'blocked')
