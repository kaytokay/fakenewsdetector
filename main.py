"""ASGI entry point; implementation lives in the truthshield package."""
from truthshield.application import app, lifespan
from truthshield.state import *
from truthshield.schemas import TextInput, URLInput
from truthshield.services.analysis import *
from truthshield.routes.api import *
