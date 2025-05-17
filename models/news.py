from pydantic import BaseModel

class NewsItem(BaseModel):
    id: int
    headline: str
    content: str
