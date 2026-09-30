import secrets
from getpass import getpass
from werkzeug.security import generate_password_hash
if __name__=='__main__':
    password=getpass('Admin password (minimum 12 characters): ')
    if len(password)<12: raise SystemExit('Choose at least 12 characters.')
    if password!=getpass('Confirm password: '): raise SystemExit('Passwords do not match.')
    print('ADMIN_PASSWORD_HASH='+generate_password_hash(password))
    print('SECRET_KEY='+secrets.token_urlsafe(48))
