from models.article import Article

def get_articles():
    return [
        Article(id=1, title='Market Update', summary='Stocks rally as inflation eases.'),
        Article(id=2, title='Tech News', summary='New AI chip released by major vendor.')
    ]
