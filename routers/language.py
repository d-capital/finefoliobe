from fastapi import APIRouter, HTTPException
from fastapi_cache.decorator import cache
from models.label import Labels, Label, ComponentsRequest
from services.language_service import get_labels_for_component

router = APIRouter()

@router.post("/", response_model=Labels)
@cache(expire=300)
async def get_component_labels(request: ComponentsRequest):
    result = get_labels_for_component(request.components, request.language)
    if not result:
        raise HTTPException(status_code=404, detail=f"Was not able to find labels for language {request.language} with requested components.")
    
    return result