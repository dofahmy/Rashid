"""Optional public-specific data refresh runner: python -m egx_public_refresh_once"""
from monitor.egx_data_refresh import refresh_daily_data

if __name__=='__main__':
    print(refresh_daily_data())
