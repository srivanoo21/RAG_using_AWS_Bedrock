"""
This code snippet uses AWS Bedrock with Boto3 to generate text (a Shakespearean poem on Generative AI) 
using a specific model
"""

# Imports and setups
import boto3
import json

# Prompt definition
prompt_data="""
Act as a Shakespeare and write a poem on Genertaive AI
"""

# Creating Bedrock Client
# A Bedrock client (bedrock-runtime) is created using Boto3
# This client allows communication with Amazon Bedrock to invoke AI models
bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Payload creation
"""
Prompt Formatting: The prompt is enclosed in [INST] and [/INST], a common format used to instruct AI models.
Generation Parameters:
max_gen_len: The maximum length of the generated text (512 tokens).
temperature: Controls randomness. A lower value (0.5) makes output more deterministic.
top_p: Controls the sampling method to pick the most likely tokens.
"""
payload={
    "prompt":"[INST]"+ prompt_data +"[/INST]",
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}
# Converts the Python dictionary payload to a JSON string.
body=json.dumps(payload)

# Model invocation
"""
Model ID: Specifies the AI model to use (meta.llama3-8b-instruct-v1:0).
invoke_model: Sends a request to AWS Bedrock to generate text based on the prompt and parameters.
body: The JSON-encoded payload.
modelId: The selected model.
accept="application/json": Specifies that the response will be in JSON format.
contentType="application/json": Specifies that the request body is in JSON format.
"""
model_id="meta.llama3-8b-instruct-v1:0"
response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

# Response parsing
response_body=json.loads(response.get("body").read())
repsonse_text=response_body['generation']
print(repsonse_text)