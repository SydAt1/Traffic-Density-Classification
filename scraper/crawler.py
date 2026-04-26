from icrawler.builtin import BingImageCrawler

crawler = BingImageCrawler(storage={"root_dir": "heavy_traffic"})
crawler.crawl(keyword="Kathmandu Kalanki traffic jam peak hour", max_num=100)