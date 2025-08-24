from fastapi import APIRouter
from models.news import NewsItem
from services.news_service import get_news

router = APIRouter()

@router.get("/")
def list_news():
    return get_news()
