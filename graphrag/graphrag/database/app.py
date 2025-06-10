import asyncio
from gremlin_python.driver import client
from gremlin_python.driver.resultset import ResultSet

client = client.Client(
    url="ws://localhost:8901/",
    traversal_source="g",
    username="/dbs/db1/colls/coll1",
    password=(
        "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnq"
        "yMsEcaGQy67XIw/Jw=="
    ),
)

async def main():
    try:
        print("After connecting")
        future = await client.execute("g.V().valueMap()")
        print("After submitting")

        rs = future.result()  
        print("After future.result")

        all_results = rs.all().result()
        print("Printing all results")
        print(all_results)
    except Exception as e:
        print(e)
    finally:
        client.close()

if __name__ == "__main__":
    asyncio.run(main())

# print("Connected")

# client.submit(message="g.V().drop()")

# print("Written")

# client.submit(
#     message=(
#         "g.addV('product').property('id', prop_id).property('name', prop_name)"
#     ),
#     bindings={
#         "prop_id": "68719518371",
#         "prop_name": "Kiama classic surfboard",
#     },
# )

# print("Written Object")
# 

