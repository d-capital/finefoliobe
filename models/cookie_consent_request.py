from pydantic import BaseModel
class CookieConsentData(BaseModel):
    """Request model for cookie consent data"""
    userId: str
    timestamp: str
    userAgent: str
