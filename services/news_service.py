from models.news import NewsItem
import feedparser
news_url = "https://investing.com/rss/news_1.rss"

def get_news():
    limit = 5
    news_feed = feedparser.parse(news_url)
    news_dict:list = list()
    for i in range(0,limit):
        news_dict.append(news_feed.entries[i])
    news_to_return = { 'news': news_dict}
    return(news_to_return)
