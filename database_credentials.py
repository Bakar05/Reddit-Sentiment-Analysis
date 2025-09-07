# database_credentials.py
from sqlalchemy import create_engine

# Database connection
username = 'root'
password = 'ab1234'
host = 'localhost'
port = 3308
database_name = 'reddit_sentimental_analysis'

engine = create_engine(f'mysql+pymysql://{username}:{password}@{host}:{port}/{database_name}')