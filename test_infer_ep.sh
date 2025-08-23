#!/bin/bash

URL="http://localhost:8000/predict"

# Array of JSON examples
EXAMPLES=(
'{"amount": 150.0, "transaction_time": 17.5, "transaction_type": "online", "location_region": "US-East"}'
'{"amount": 900, "transaction_time": 2.0, "transaction_type": "online", "location_region": "EU"}'
'{"amount": 20.0, "transaction_time": 9.0, "transaction_type": "in-store", "location_region": "Asia"}'
'{"amount": 300.0, "transaction_time": 22.0, "transaction_type": "online", "location_region": "US-West"}'
)

# Number of requests
NUM_REQUESTS=200

for ((i=1;i<=NUM_REQUESTS;i++)); do
  # Decide whether this is fraud (10% chance)
  if (( RANDOM % 10 == 0 )); then
    JSON='{"amount": 5000.0, "transaction_time": 2.0, "transaction_type": "online", "location_region": "Asia"}'  # fraud
  else
    # Pick a random normal example
    JSON=${EXAMPLES[$((RANDOM % ${#EXAMPLES[@]}))]}
  fi

  echo "Request #$i: $JSON"
  curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$JSON"
  echo -e "\n---"
  sleep 1
done
