from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
user = "neo4j"
password = "RSDprocessingKG"  # 你记事本里的密码

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session(database="neo4j") as session:
    v = session.run("RETURN 1 AS ok").single()["ok"]
    print("Neo4j OK =", v)
driver.close()
