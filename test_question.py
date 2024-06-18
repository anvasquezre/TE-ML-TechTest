import requests

url = "http://127.0.0.1:8000/api/query"
data = {
    "question": "Has Lorena Valencia bought any property?"
}

response = requests.post(url, json=data)

# Print the raw response text
print("Raw response:", response.text)

try:
    # Try to parse the response as JSON
    json_response = response.json()
    print("JSON response:", json_response)
except requests.exceptions.JSONDecodeError as e:
    print("Failed to decode JSON response:", e)

