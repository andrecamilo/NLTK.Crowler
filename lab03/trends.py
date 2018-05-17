from autoriza_twitter import api
import json

trends = api.trends_place(23424768)
print(json.dumps(trends, indent=1))
