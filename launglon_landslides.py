import os
import requests

# 1. Load Credentials
with open('credentials_CDSE.txt', 'r') as f:
    lines = f.read().splitlines()
    email = lines[0].strip()
    password = lines[1].strip()

# 2. Hardcoded WKT Footprint (Space removed after POLYGON for API compliance)
wkt_footprint = "POLYGON((98.03296553941765 13.52539483662376, 98.25962605013193 13.52430105893067, 98.2447006633641 14.23283538940989, 98.02936557680786 14.2317630713818, 98.03296553941765 13.52539483662376))"

# 3. Authenticate with CDSE Identity Provider
print("Authenticating with CDSE...")
token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
token_data = {
    "client_id": "cdse-public",
    "username": email,
    "password": password,
    "grant_type": "password"
}
auth_response = requests.post(token_url, data=token_data)
auth_response.raise_for_status()
access_token = auth_response.json()['access_token']
headers = {"Authorization": f"Bearer {access_token}"}

# 4. Define Search Parameters
odata_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

time_windows = {
    "pre_event": ("2026-09-15T00:00:00.000Z", "2026-09-25T23:59:59.000Z"),
    "post_event": ("2026-09-27T00:00:00.000Z", "2026-10-08T23:59:59.000Z")
}

def search_products(start_date, end_date):
    # OData filter using 'area=' and the pre-cleaned WKT string
    query_filter = (
        f"Collection/Name eq 'SENTINEL-1' and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD') and "
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'sensorOperationalMode' and att/OData.CSC.StringAttribute/Value eq 'IW') and "
        f"ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date} and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt_footprint}')"
    )
    
    params = {
        "$filter": query_filter,
        "$orderby": "ContentDate/Start desc",
        "$top": 5
    }
    
    response = requests.get(odata_url, params=params)
    response.raise_for_status()
    return response.json().get('value', [])

# 5. Search and Download
for event_phase, (start, end) in time_windows.items():
    print(f"\nSearching {event_phase} window ({start} to {end})...")
    products = search_products(start, end)
    
    if not products:
        print(f"No products found for {event_phase}. (If post-event is missing, it is still processing at Copernicus).")
        continue

    for product in products:
        prod_id = product['Id']
        prod_name = product['Name']
        
        print(f"Found: {prod_name}")
        download_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({prod_id})/$value"
        
        output_file = f"{prod_name}.zip"
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping download.")
            continue
            
        print(f"Downloading {output_file}...")
        with requests.get(download_url, headers=headers, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            with open(output_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")