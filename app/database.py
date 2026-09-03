from pymongo import MongoClient

from app.config import MONGO_URI, MONGO_DB_NAME

# Single shared connection, reused across requests and scripts
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[MONGO_DB_NAME]

# Separate connection -- Spring Boot's live category/product data lives here,
# on a different Mongo database (catalog) on the same 192.168.0.109 instance.
CATALOG_MONGO_URI = "mongodb://192.168.0.109:27017"
CATALOG_DB_NAME = "catalog"

catalog_client = MongoClient(CATALOG_MONGO_URI)
catalog_db = catalog_client[CATALOG_DB_NAME]