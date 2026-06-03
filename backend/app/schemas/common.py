from pydantic import BaseModel


class IDOnly(BaseModel):
    id: str


class MessageResponse(BaseModel):
    message: str
