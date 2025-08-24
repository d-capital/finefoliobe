from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import news, exchange, macro_data, countries
from contextlib import asynccontextmanager
from db.session import init_db
from jobs.macro_update import update

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
#scheduler.add_job(update, 'interval', minutes=15)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler.start()
    yield


app = FastAPI(title="Finance API", lifespan=lifespan)

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
app.include_router(countries.router, prefix="/countries")
