
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_predict_simple():
    print("\n--- Testing /predict (Simple) ---")
    payload = {"price": 100.0}
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        response.raise_for_status()
        data = response.json()
        print("Success!")
        print(json.dumps(data, indent=2))
        
        # Validation
        assert "predicted_demand" in data
        assert "predicted_revenue" in data
        assert "predicted_profit" in data
        assert data["predicted_revenue"] == data["predicted_demand"] * 100.0
        
    except Exception as e:
        print(f"FAILED: {e}")
        if 'response' in locals():
            print(response.text)
        sys.exit(1)

def test_predict_full():
    print("\n--- Testing /predict (Full) ---")
    payload = {
        "price": 100.0,
        "cost": 50.0,
        "competitor_price": 95.0,
        "discount": 0.1,
        "elasticity_index": 1.2,
        "return_rate": 2.5,
        "reviews": 4.2
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        response.raise_for_status()
        data = response.json()
        print("Success!")
        print(json.dumps(data, indent=2))
        
        # Validation
        demand = data["predicted_demand"]
        expected_revenue = demand * 100.0
        expected_profit = demand * (100.0 - 50.0)
        
        # Float comparison
        assert abs(data["predicted_revenue"] - expected_revenue) < 0.01
        assert abs(data["predicted_profit"] - expected_profit) < 0.01
        
    except Exception as e:
        print(f"FAILED: {e}")
        if 'response' in locals():
            print(response.text)
        sys.exit(1)

def test_sensitivity():
    print("\n--- Testing /sensitivity ---")
    params = {"price": 100.0, "cost": 50.0}
    try:
        response = requests.get(f"{BASE_URL}/sensitivity", params=params)
        response.raise_for_status()
        data = response.json()
        print("Success!")
        # Print summary instead of full arrays
        print(f"Prices count: {len(data['prices'])}")
        print(f"Revenues count: {len(data['revenues'])}")
        print(f"Profits count: {len(data['profits'])}")
        
        assert len(data['prices']) == 20
        assert len(data['revenues']) == 20
        assert len(data['profits']) == 20
        
    except Exception as e:
        print(f"FAILED: {e}")
        if 'response' in locals():
            print(response.text)
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_predict_simple()
        test_predict_full()
        test_sensitivity()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST SCRIPT FAILED: {e}")
        sys.exit(1)
