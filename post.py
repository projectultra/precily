import requests

api_url = "http://127.0.0.1:5000/api/endpoint"

payload = {
    "text1": "money money money",
    "text2": "money money money"
}
response = requests.post(api_url, json=payload)
if response.status_code == 200:
    # Request was successful
    result = response.json()
    print("Similarity Score:", result['similarity score'])
else:
    # Request was unsuccessful
    print("Error:", response.text)