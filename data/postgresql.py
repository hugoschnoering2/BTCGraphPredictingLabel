
import json
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool


def preprocessing(v):
    if isinstance(v, str):
        return "'" + v + "'"
    elif isinstance(v, datetime.datetime):
        return "'" + v.isoformat(sep=" ") + "'"
    elif isinstance(v, dict):
        return "'" + json.dumps(v) + "'"
    elif isinstance(v, bytes):
        return str(psycopg2.Binary(v))
    elif v is None:
        return "NULL"
    else:
        return str(v)


def conditions_to_str(conditions: list):
    if conditions is None or len(conditions) == 0:
        return ""
    elif len(conditions) == 1:
        return " WHERE " + str(conditions[0])
    else:
        return " WHERE " + " AND ".join(list(map(str, conditions)))


class Condition:

    def __init__(self, col: str, ct: str, value):
        self.col = col
        self.ct = ct
        self.value = value

    def __str__(self):
        if self.ct in ["=", "<=", ">=", "<", ">", "!=", "NOT"]:
            return f"{self.col} {self.ct} {preprocessing(self.value)}"
        elif self.ct == "IN":
            if len(self.value) > 1:
                return f"{self.col} IN ({','.join([preprocessing(v) for v in self.value])})"
            else:
                return f"{self.col} = {preprocessing(self.value[0])}"
        else:
            raise ValueError


class PostgresqlDataService:

    def __init__(self, endpoint: str, user: str, password: str, port: int = 5432, db: str = "postgres"):

        self.endpoint = endpoint
        self.user = user
        self.password = password
        self.port = port
        self.db = db

    @property
    def connector(self):
        conn_str="host={0} dbname={1} user={2} password={3} port={4}".format(
                  self.endpoint, self.db, self.user, self.password, self.port)
        return psycopg2.connect(conn_str)

    def pool(self, min_connection: int, max_connection: int):
        return ThreadedConnectionPool(minconn=min_connection, maxconn=max_connection,
                                      user=self.user, password=self.password, host=self.endpoint,
                                      database=self.db, port=self.port)

    def execute_query(self, query: str, fetch: str = None):
        connector = self.connector
        cursor = connector.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        if fetch is None:
            resp = None
        elif fetch == "all":
            resp = cursor.fetchall()
        elif fetch == "one":
            resp = cursor.fetchone()
        else:
            raise ValueError
        connector.commit()
        connector.close()
        return resp

    @classmethod
    def execute_query_w_connector(cls, connector, query: str, fetch: str = None):
        cursor = connector.cursor(cursor_factory=RealDictCursor)
        cursor.execute(query)
        if fetch is None:
            resp = None
        elif fetch == "all":
            resp = cursor.fetchall()
        elif fetch == "one":
            resp = cursor.fetchone()
        else:
            raise ValueError
        cursor.close()
        return resp

    def fetch(self, table: str, columns: list = None, conditions: list = [],
              orderby: str = None, order: str = "asc", limit: int = None, distinct: bool = False):
        connector = self.connector
        cursor = connector.cursor(cursor_factory=RealDictCursor)
        columns = ",".join(columns) if columns is not None else "*"
        r = f"SELECT DISTINCT {columns} FROM {table}" if distinct else f"SELECT {columns} FROM {table}"
        r += conditions_to_str(conditions)
        if orderby is not None: r += f" ORDER BY {orderby} {order}"
        if limit is not None: r += f" LIMIT {limit}"
        try:
            cursor.execute(r)
            resp = cursor.fetchall()
            connector.close()
            return resp
        except Exception as e:
            connector.close()
            raise e
