from pydantic import BaseModel

class Label(BaseModel):
    language: str
    component: str
    label: str
    text: str
    
class Labels(BaseModel):
    labels: list[Label]

class ComponentsRequest(BaseModel):
    language: str
    components: list[str]