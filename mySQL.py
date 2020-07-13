from mysql import connector
import os

class MySQL:
    def __init__(self):
        self.connection = connector.connect(host=os.getenv('DB_HOST'), user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'), database=os.getenv('DB_NAME'))
        self.cursor = self.connection.cursor()

    def query(self, queryString):
        self.cursor.execute(queryString)
        return self.cursor.fetchall()

    def write(self, writeString):
        self.cursor.execute(writeString)
        self.connection.commit()
    def disconnect(self):
        self.connection.close()