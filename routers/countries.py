from fastapi import APIRouter, HTTPException,  Depends
from models.country import Country, CountriesRequest
from db.session import get_session
from sqlmodel import Session
from repositories.macrodata import MacroDataRepository

router = APIRouter()

@router.post("/",response_model=list[Country])
def countries(countries:CountriesRequest, session: Session = Depends(get_session)):
    result = MacroDataRepository(session).get_countries_last_events(countries.codes)
    if not result:
        raise HTTPException(status_code=404, detail="Macro event for the given country was not found.")
    return result
