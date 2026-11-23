"""SQLite is the durable source; wakeup queues are coalesced notifications only."""
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from embodied_harness.domain.models import uid


class EventStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE, kind TEXT, ts REAL, payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS cursors(consumer TEXT PRIMARY KEY, seq INTEGER)")
        self.wakeups = {}

    def emit(self, kind, payload, event_id=None):
        with self.db:
            row = self.db.execute("INSERT OR IGNORE INTO events(id,kind,ts,payload) VALUES(?,?,?,?)",
                                  (event_id or uid(), kind, time.time(), json.dumps(payload)))
        if row.rowcount:
            for event in self.wakeups.values():
                event.set()
        return row.lastrowid

    def since(self, cursor):
        return [{"seq": r[0], "kind": r[1], "payload": json.loads(r[2])}
                for r in self.db.execute("SELECT seq,kind,payload FROM events WHERE seq>? ORDER BY seq", (cursor,))]

    async def stream(self, consumer):
        wakeup = self.wakeups.setdefault(consumer, asyncio.Event())
        row = self.db.execute("SELECT seq FROM cursors WHERE consumer=?", (consumer,)).fetchone()
        cursor = row[0] if row else 0
        while True:
            wakeup.clear()
            rows = self.since(cursor)
            if not rows:
                await wakeup.wait()
                continue
            for event in rows:
                yield event
                cursor = event["seq"]
                with self.db:
                    self.db.execute("INSERT OR REPLACE INTO cursors VALUES(?,?)", (consumer, cursor))

    def close(self):
        self.db.close()
