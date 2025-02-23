"""
This code snippet uses AWS Bedrock with Boto3 to generate text (a Shakespearean poem on Generative AI) 
using a specific model
"""

# Imports and setups
import boto3
import json

# Define the prompt
prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

# Creating Bedrock Client
# A Bedrock client (bedrock-runtime) is created using Boto3
# This client allows communication with Amazon Bedrock to invoke AI models
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")


# Payload Preparation
"""
payload: Specifies the input parameters for the AI model:

"prompt": The text input to the model.
"max_tokens": Limits the number of tokens (words/pieces of text) the model can generate.
"temperature": Controls randomness. A higher value (e.g., 0.8) produces more creative output, while lower values make it more deterministic.
"top_p": Another randomness parameter; restricts token selection to the most probable tokens summing to 0.8.
"""
payload={
    "messages": [
   {
    "role": "user",
    "content": prompt_data
   }
   ],
    "max_tokens":512,
    "temperature":0.8,
    "top_p":0.8
}

# Below line converts the payload dictionary to a JSON string
body = json.dumps(payload)


# Invoke the model
"""
model_id: Specifies the foundation model to use (ai21.j2-mid-v1 in this case).
invoke_model: Sends a request to the model with the provided prompt and configuration.
accept and contentType: Indicate that the input and output should be in JSON format.
"""

model_id = "ai21.jamba-1-5-mini-v1:0"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)


# Extracting the Response
"""
response_body: Parses the JSON response from the model.
response_text: Extracts the generated text from the response. The output is located in completions[0].data.text (this may vary by model).
print(response_text): Displays the generated poem on the console.
"""
response_body = json.loads(response.get("body").read())
response_text = response_body.get("choices")[0].get("message").get("content")
print(response_text)