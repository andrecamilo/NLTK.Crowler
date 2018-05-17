idle3 crawler.py

import crawler
from crawler import AppCrawler
novo_crawler = AppCrawler("http://www.uol.com", 1)
novo_crawler.crawl()

for app in novo_crawler.apps:
    print(app)
