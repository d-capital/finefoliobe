from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import news, exchange

app = FastAPI(title="Finance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://finefoliobe.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news.router, prefix="/news")
app.include_router(exchange.router, prefix="/exchange")
