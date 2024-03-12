import json
import requests


# get any user from DB....

target_user_name = 'Francis Robinson'
genre = 'Drama,Romance'
country = 'South Korea'


data = {
        'name':target_user_name,
        'genre':genre,
        'country':country
        }


json_data = json.dumps(data)


# The URL to send the data to
url = 'http://127.0.0.1:8000/matchuser'

# Send the data as a POST request
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# Check response status (200 is OK)
print(response.status_code)

if response.status_code == 200:


    # Print response data
    print(response.text)

else:
    print('failure')
