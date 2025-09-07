# database_credentials.py
from sqlalchemy import create_engine

# Database connection
username = ''
password = ''
host = ''
port = 
database_name = ''


engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}')
