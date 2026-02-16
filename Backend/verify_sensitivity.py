import requests
import json

URL = "http://127.0.0.1:8000/sensitivity"

params = {
    "price": 100.0,
    "cost": 50.0
}

try:
    response = requests.get(URL, params=params)
    if response.status_code == 200:
        data = response.json()
        print("Success! Endpoint returned 200 OK.")
        print(f"Keys in response: {list(data.keys())}")
        if "plot_json" in data:
            print("plot_json found in response.")
            # Verify it's valid JSON
            plot_data = json.loads(data["plot_json"])
            print("plot_json is valid JSON structure.")
        else:
            print("ERROR: plot_json MISSING from response.")
    else:
        print(f"Failed with status code: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
