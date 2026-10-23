"""Request payload schemas."""
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str


class URLInput(BaseModel):
    url: str

