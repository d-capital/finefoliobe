import csv
from datetime import datetime
import os
from fastapi import APIRouter, HTTPException
from models.cookie_consent_request import CookieConsentData

router = APIRouter()

# Settings for cookie consent data
CONSENT_DATA_DIR = os.getenv('CONSENT_DATA_DIR', 'cookie_consent_data')
CONSENT_DATA_FILE = os.path.join(CONSENT_DATA_DIR, 'cookie_consents.csv')
# Create directory if it doesn't exist
os.makedirs(CONSENT_DATA_DIR, exist_ok=True)
# Initialize CSV file with headers
if not os.path.exists(CONSENT_DATA_FILE):
    with open(CONSENT_DATA_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['userId', 'timestamp', 'userAgent', 'receivedAt'])

@router.post("/")
async def save_cookie_consent(data: CookieConsentData):
    try:
        # Validate data
        if not data.userId or not data.timestamp:
            raise HTTPException(status_code=400, detail='Missing required fields')
        
        # Get current time
        received_at = datetime.now().isoformat()
        
        # Save to CSV
        with open(CONSENT_DATA_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                data.userId,
                data.timestamp,
                data.userAgent or 'Unknown',
                received_at
            ])
        print(CONSENT_DATA_DIR)
        # Log the consent
        print(f'[{received_at}] Cookie consent saved for user: {data.userId}')
        
        return {
            'success': True,
            'message': 'Cookie consent recorded successfully'
        }
    
    except Exception as e:
        print(f'Error saving cookie consent: {str(e)}')
        raise HTTPException(status_code=500, detail=f'Failed to save cookie consent: {str(e)}')