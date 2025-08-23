from huggingface_hub import InferenceClient

client = InferenceClient(model="gpt2")
response = client.text_generation("Hello, world!")
print(response.generated_text)