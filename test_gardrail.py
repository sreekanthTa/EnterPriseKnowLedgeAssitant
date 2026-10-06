from dotenv import load_dotenv
from nemoguardrails import RailsConfig, LLMRails


load_dotenv()
config = RailsConfig.from_path("./guardrails")

rails = LLMRails(config)

response = rails.generate(
    messages=[
        {
            "role": "user",
            "content": "hello"
        }
    ]
)

print(response)