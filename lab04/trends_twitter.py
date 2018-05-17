# -*- coding: utf-8 -*-
from autoriza_app import api
import json

WOEID = 23424768   # WOEID do Brasil
trends = api.trends_place(WOEID)

for trend in trends[0]['trends']:
	print(trend["name"])