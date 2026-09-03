from pymongo import MongoClient

c = MongoClient('mongodb://192.168.0.109:27017')

doc = c['catalog'].categories.find_one()
print('sample doc:')
print(doc)
print()

print('all field names seen across a few docs:')
fields = set()
for d in c['catalog'].categories.find().limit(20):
    fields.update(d.keys())
print(fields)