#!/usr/bin/env python3
"""
Test Genability API authentication
"""
import requests
from requests.auth import HTTPBasicAuth

# Your current credentials
app_id = "f13be756-3006-47df-b4b5-854cd0e7f0a7"
app_key = "8dc901aa-41d7-449b-a7f5-2cba19624cea"

# Test authentication
auth = HTTPBasicAuth(app_id, app_key)
url = "https://api.genability.com/rest/v1"


try:
    response = requests.get(url, auth=auth, timeout=10)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Authentication successful!")
        print(f"Response: {response.json()}")
    elif response.status_code == 401:
        print("❌ Authentication failed: Invalid credentials")
        print("Please check your App ID and App Key")
    else:
        print(f"❌ Unexpected error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Connection error: {str(e)}")