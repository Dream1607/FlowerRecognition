import MySQLdb

class DB(object):

    def __init__(self, db_name):
        try:
            self.conn = MySQLdb.connect(host='localhost',user='root',passwd='',db=db_name,charset="utf8")
        except Exception, e:
            print e

        self.cursor=self.conn.cursor()
        self.cursor.execute("SET NAMES utf8")

    def getOneNumber(self, query):
        self.cursor.execute(query)
        a = self.cursor.fetchall()
        return a[0][0]

    def select(self, query):
        try:
            r = self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception, e:
            print e

    # insert a list with column_names list
    def insert(self, table_name, column_names, data):
        if len(column_names)!= len(data[0]):
            print "The number of columns_names should be equal to data!"
            return
        try:
            query = "INSERT INTO "+table_name+" ("
            query += ",".join(column_names)
            query += ") VALUES ("
            s = []
            for i in range(0,len(column_names)):
                s.append("%s")
            query += ",".join(s)
            query += ")"
            self.cursor.executemany(query, data)
        except Exception, e:
            print e
        self.conn.commit()

    def query(self, query):
        try:
            r = self.cursor.execute(query)
            return r
        except Exception, e:
            print e

    def closeDB(self):
        self.cursor.close()
        self.conn.close()