from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import news, exchange, macro_data, countries, valuation, screener
from contextlib import asynccontextmanager
from db.session import init_db
from jobs.macro_update import update

from apscheduler.schedulers.background import BackgroundScheduler

scheduler = BackgroundScheduler()
#scheduler.add_job(update, 'interval', minutes=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    #scheduler.start()
    yield


app = FastAPI(title="Finance API", lifespan=lifespan)

origins = [
    "https://localhost",        # Если вы заходите по https на локалке
    "https://fine-folio.ru",   # Ваш реальный домен
    "http://localhost",         # Если используется http
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

from fastapi.routing import APIRoute

print("\nRegistered routes:")
for route in app.routes:
    if isinstance(route, APIRoute):
        print(f"{route.methods} {route.path}")