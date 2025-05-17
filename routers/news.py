from fastapi import APIRouter
from models.news import NewsItem
from news_service import get_news

router = APIRouter()

@router.get("/", response_model=list[NewsItem])
def list_news():
    return get_news()
