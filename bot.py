import os, time, logging
import requests
import json
from pathlib import Path
from sqlalchemy import select
from core import database, Setting, Outbox, Lead, handle_update, now
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
DB=database()
TOKEN=os.getenv('BOT_TOKEN','')

def api(method,payload):
    try:
        payload=dict(payload)
        if method=='sendPhoto' and payload.pop('_welcome_photo',False):
            photo=Path(__file__).resolve().parent/'assets'/'rashid_welcome.png'
            if not photo.is_file():
                # Keep registration working even if the optional asset was omitted.
                payload['text']=payload.pop('caption')
                return api('sendMessage',payload)
            form={k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in payload.items()}
            with photo.open('rb') as stream:
                r=requests.post(f'https://api.telegram.org/bot{TOKEN}/sendPhoto',data=form,files={'photo':('rashid_welcome.png',stream,'image/png')},timeout=45)
                return r.json()
        r=requests.post(f'https://api.telegram.org/bot{TOKEN}/{method}',json=payload,timeout=45)
        return r.json()
    except (requests.RequestException,ValueError):
        return {'ok':False,'error_code':0,'description':'Network error'}

def deliver():
    import json
    with DB() as s:
        rows=s.scalars(select(Outbox).where(Outbox.status=='pending',Outbox.next_at<=int(time.time())).order_by(Outbox.id).limit(30)).all()
        for row in rows:
            # Preserve per-chat order while a previous message awaits retry.
            earlier=s.scalar(select(Outbox.id).where(Outbox.chat_id==row.chat_id,Outbox.id<row.id,Outbox.status=='pending').limit(1))
            if earlier: continue
            result=api(row.method,json.loads(row.payload)); row.attempts+=1
            if result.get('ok'):
                row.status='sent'; row.error=''
                if row.method in ('sendMessage','sendPhoto'):
                    l=s.scalar(select(Lead).where(Lead.telegram_id==row.chat_id))
                    if l: l.telegram_status='active'; l.last_message_at=now()
            else:
                code=result.get('error_code',0); row.error=f'Telegram error {code}'
                if code==429:
                    row.next_at=int(time.time())+int(result.get('parameters',{}).get('retry_after',30))
                elif code in (400,403):
                    row.status='failed'
                    if code==403:
                        l=s.scalar(select(Lead).where(Lead.telegram_id==row.chat_id))
                        if l: l.telegram_status='blocked'
                else: row.next_at=int(time.time())+min(300,2**min(row.attempts,8))
            s.commit()
            time.sleep(.06)

def main():
    if not TOKEN: raise SystemExit('BOT_TOKEN is required')
    ids=[v.strip() for v in os.getenv('ADMIN_TELEGRAM_IDS','').split(',') if v.strip()]
    if any(not v.isdecimal() or len(v)>18 for v in ids): raise SystemExit('ADMIN_TELEGRAM_IDS must contain numeric IDs separated by commas.')
    me=api('getMe',{})
    if not me.get('ok'): raise SystemExit('Unable to connect to Telegram. Check BOT_TOKEN and network.')
    hook=api('getWebhookInfo',{})
    if not hook.get('ok') or hook.get('result',{}).get('url'): raise SystemExit('Polling requires an empty webhook. Use a dedicated new bot or remove its webhook first.')
    logging.info('Bot connected. Run exactly one bot worker.')
    while True:
        try:
            deliver()
            with DB() as s:
                state=s.get(Setting,'offset'); offset=int(state.value) if state else 0
            result=api('getUpdates',{'offset':offset,'timeout':20,'allowed_updates':['message','callback_query']})
            if not result.get('ok'):
                logging.warning('Telegram polling error %s',result.get('error_code',0)); time.sleep(5); continue
            for update in result['result']:
                with DB.begin() as s:
                    handle_update(s,update)
                    state=s.get(Setting,'offset')
                    if not state: state=Setting(key='offset'); s.add(state)
                    state.value=str(update['update_id']+1)
                deliver()
        except Exception:
            # Avoid printing tokens, customer data, or HTTP URLs in logs.
            logging.error('Worker error; progress retained. Retrying in 5 seconds.'); time.sleep(5)
if __name__=='__main__': main()
