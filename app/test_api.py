# app/test_api.py
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health Check: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"\nModel Info: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction"""
    url = f"{BASE_URL}/predict"
    
    sample_house = {
        "area_sqft": 1500,
        "bedroom_num": 3,
        "bathroom_num": 2,
        "balcony_num": 2,
        "property_type": "flat",
        "facing": "North",
        "agePossession": "5-10 years",
        "floor_number": 3,
        "total_floors": 10,
        "feature_count": 5,
        "furnish_count": 3,
        "safety_rating": 4.5,
        "lifestyle_rating": 4.0,
        "green_area_rating": 3.5,
        "amenities_rating": 4.2,
        "rate_per_sqft": 5000
    }
    
    response = requests.post(url, json=sample_house)
    print(f"\nSingle Prediction: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Predicted Price: {data['predicted_price_lakhs']} lakhs ({data['predicted_price_crores']} crores)")
        print(f"Model Version: {data['model_version']} ({data['model_stage']})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction"""
    url = f"{BASE_URL}/predict/batch"
    
    batch_houses = {
        "houses": [
            {
                "area_sqft": 1200,
                "bedroom_num": 2,
                "bathroom_num": 2,
                "balcony_num": 1,
                "property_type": "flat",
                "facing": "East",
                "agePossession": "3-5 years",
                "floor_number": 2,
                "total_floors": 5,
                "feature_count": 4,
                "furnish_count": 2,
                "safety_rating": 4.0,
                "lifestyle_rating": 3.8,
                "green_area_rating": 3.2,
                "amenities_rating": 3.9,
                "rate_per_sqft": 4500
            },
            {
                "area_sqft": 2500,
                "bedroom_num": 4,
                "bathroom_num": 3,
                "balcony_num": 3,
                "property_type": "house",
                "facing": "South",
                "agePossession": "1-3 years",
                "floor_number": 1,
                "total_floors": 2,
                "feature_count": 8,
                "furnish_count": 5,
                "safety_rating": 4.8,
                "lifestyle_rating": 4.5,
                "green_area_rating": 4.2,
                "amenities_rating": 4.6,
                "rate_per_sqft": 6000
            }
        ]
    }
    
    response = requests.post(url, json=batch_houses)
    print(f"\nBatch Prediction: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total Predictions: {data['total_predictions']}")
        print(f"Average Price: {data['average_price_lakhs']} lakhs")
        print(f"Price Range: {data['min_price_lakhs']} - {data['max_price_lakhs']} lakhs")
        
        for i, pred in enumerate(data['predictions'], 1):
            print(f"  House {i}: {pred['predicted_price_lakhs']} lakhs")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("=" * 50)
    print("Testing House Price Prediction API")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")