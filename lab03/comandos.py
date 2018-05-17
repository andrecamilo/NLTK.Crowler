idle3 autoriza_twitter.py

from autoriza_twitter import api

trends = api.trends_place(23424768)
print(trends)
import json
print(json.dumps(trends, indent=1))
for trend in trends[0]["trends"]:
    print(trend["name"])

python2.7 trends.py
python2.7 trends.py > ttbrasil.txt
