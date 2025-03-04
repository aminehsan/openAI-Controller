import openai


class OpenAIController:
    def __init__(self, api_key: str = None, **kwargs):
        """A Python class for seamless interaction with OpenAI's API, supporting GPT and other LLM models."""

        self._client = openai.OpenAI(api_key=api_key, **kwargs)

    def text_generator(self, model: str, system_role_content: str, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        completion = self._client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': system_role_content
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            **kwargs
        )
        return completion.choices[0].message.content.strip()
