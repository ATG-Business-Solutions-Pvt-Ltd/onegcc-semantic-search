from pydantic import BaseModel

class AskRequest(BaseModel):
    prompt: str

class PromptCreate(BaseModel):
    content: str 

class SearchQuery(BaseModel):
    query: str