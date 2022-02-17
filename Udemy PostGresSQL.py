import psycopg2
def create_table():
    conn = sqlite3.connect("lite.db")
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS STORE (ITEM TEXT, QUANTITY INTEGER, PRICE REAL)")
    conn.commit()
    conn.close() 

def insert(item,quantity,price):
    conn = sqlite3.connect("lite.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO STORE VALUES (?,?,?)",(item,quantity,price))
    conn.commit()
    conn.close() 

def view():
    conn = sqlite3.connect("lite.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM STORE")
    rows = cur.fetchall()
    conn.close() 
    return rows

def delete(item):
    conn = sqlite3.connect("lite.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM STORE WHERE ITEM=?",(item,))
    conn.commit()
    conn.close() 

def update(item,quantity,price):
    conn = sqlite3.connect("lite.db")
    cur = conn.cursor()
    cur.execute("UPDATE STORE SET QUANTITY=?,PRICE=? WHERE ITEM=?", (quantity,price,item,))
    conn.commit()
    conn.close() 

insert('Wine Glass',10,5)
print(view())
#delete('Wine Glass')
print(view())
update('Wine Glass',10,15)
print(view())