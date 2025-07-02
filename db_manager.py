
import time, json, hashlib, sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from core.config import DATABASE_URL, CACHE_TTL_SECONDS
engine = sa.create_engine(DATABASE_URL, future=True)
meta = sa.MetaData()
cache = sa.Table('cache', meta,
    sa.Column('hash', sa.String, primary_key=True),
    sa.Column('payload', sa.Text),
    sa.Column('expires_at', sa.Float))
meta.create_all(engine)
Session = sessionmaker(engine, expire_on_commit=False)
class DBManager:
    def __init__(self):
        self.session = Session()
    def _h(self, key):
        return hashlib.sha256(key.encode()).hexdigest()
    def get(self, key):
        row = self.session.get(cache, self._h(key))
        if row and row.expires_at > time.time():
            return json.loads(row.payload)
        return None
    def set(self, key, value, ttl=CACHE_TTL_SECONDS):
        h = self._h(key)
        with self.session.begin():
            self.session.merge({'hash': h,'payload': json.dumps(value),'expires_at': time.time()+ttl})
