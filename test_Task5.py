from fastapi.testclient import TestClient
from Task5 import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_read_predict_volcano():
    response = client.post("/predict/", json={"text": "volcano"} )
    json_data = response.json()
    assert response.status_code == 200
    assert json_data == 980

    
def test_read_predict_ocean():
    response = client.post("/predict/", json={"text": "ocean"} )
    json_data = response.json()
    assert response.status_code == 200
    assert json_data == 392

