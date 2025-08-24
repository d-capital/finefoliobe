from models.news import NewsItem
import feedparser
news_url = "https://investing.com/rss/news_1.rss"
news_url_ru = "https://ru.investing.com/rss/news_1.rss"

def get_news():
    limit = 5
    news_feed = feedparser.parse(news_url)
    news_feed_ru = feedparser.parse(news_url_ru)
    news_dict:list = list()
    for i in range(0,limit):
        news_dict.append(news_feed.entries[i])
    news_dict_ru: list = list()
    for i in range(0,limit):
        news_dict_ru.append(news_feed_ru.entries[i])
    news_to_return = { 'news': news_dict, 'news_ru': news_dict_ru}
    return(news_to_return)
