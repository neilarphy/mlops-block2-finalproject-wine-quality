def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_path" in data
    assert "features" in data
    assert isinstance(data["features"], list)
