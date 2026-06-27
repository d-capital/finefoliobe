from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import news, exchange, macro_data, countries, saveconsent, valuation, screener
from contextlib import asynccontextmanager
from db.session import init_db
from jobs.macro_update import update
from jobs.local_cache import cache_nyse_data, cache_nasdaq_data, cache_moex_data
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from pytz import timezone

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
moscow_tz = timezone('Europe/Moscow')
scheduler.add_job(cache_nasdaq_data, 'cron',hour=5,minute=0,timezone=moscow_tz,id='daily_5am_russia_nasdaq')
scheduler.add_job(cache_nyse_data, 'cron',hour=5,minute=0,timezone=moscow_tz,id='daily_5am_russia_nyse')
scheduler.add_job(cache_moex_data,  'cron',hour=5,minute=0,timezone=moscow_tz,id='daily_5am_russia_moex')

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    FastAPICache.init(InMemoryBackend())
    scheduler.start()
    yield


app = FastAPI(title="Finance API", lifespan=lifespan)

origins = [
    "https://localhost",        # Если вы заходите по https на локалке
    "https://fine-folio.ru",   # Ваш реальный домен
    "http://localhost:4200",         # Если используется http
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news.router, prefix="/news")
app.include_router(exchange.router, prefix="/exchange")
app.include_router(macro_data.router, prefix="/macro_data")
app.include_router(countries.router, prefix="/countries")
app.include_router(valuation.router,prefix="/valuation")
app.include_router(screener.router, prefix="/screener")
app.include_router(saveconsent.router, prefix="/saveconsent")

from fastapi.routing import APIRoute

print("\nRegistered routes:")
for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"{route.methods} {route.path}")