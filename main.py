from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import news, exchange, macro_data

app = FastAPI(title="Finance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news.router, prefix="/news")
app.include_router(exchange.router, prefix="/exchange")
app.include_router(macro_data.router, prefix="/macro_data")
