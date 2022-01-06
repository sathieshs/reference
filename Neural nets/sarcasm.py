import json

with open('sarcasm.json') as f:
    content=f.load()

labels=[]
links=[]
headlines=[]

for item in content:
    labels.append(item['label'])
    links.append(item['link'])
    headlines.append(item['headline'])

