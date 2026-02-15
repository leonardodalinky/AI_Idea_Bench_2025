import json
import os
import re
import time
from pathlib import Path
from random import choice
from typing import List

from openai import OpenAI


class Deepseek:
    def __init__(
        self,
        api_key_list_deepseek,
        base_url_deepseek,
        model_name_deepseek="deepseek-chat",
        temperature_deepseek=0.6,
    ):

        self.api_key_list_deepseek = api_key_list_deepseek
        self.base_url_deepseek = base_url_deepseek

        self.model_name_deepseek = model_name_deepseek
        self.temperature_deepseek = temperature_deepseek

    def send_request(self, prompt, question):
        """ """
        output = {}
        while True:
            try:

                api_key_deepseek = choice(self.api_key_list_deepseek)

                client_deepseek = OpenAI(api_key=api_key_deepseek, base_url=self.base_url_deepseek)

                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ]

                if "gpt" in self.model_name_deepseek:
                    completion_params = {
                        "max_completion_tokens": 8000,
                        "reasoning_effort": "low",
                    }
                else:
                    completion_params = {
                        "max_tokens": 8000,
                        "temperature": self.temperature_deepseek,
                        "response_format": {"type": "json_object"},
                    }
                output = client_deepseek.chat.completions.create(
                    model=self.model_name_deepseek,
                    messages=messages,
                    stream=False,
                    **completion_params,
                )
                output = self.postprocess(output)

                break
            except Exception as e:
                print("Exception:", e)

        return output

    def forward(self, prompt, question) -> str:
        """ """

        model_output = self.send_request(prompt, question)

        # time.sleep(2)
        return model_output

    def postprocess(self, output):
        """ """
        model_output = None
        if isinstance(output, str):
            model_output = output
        elif "gpt" in self.model_name_deepseek:
            content = output.choices[0].message.content
            # extract ```json``` block
            json_match = re.search(
                r"(?:```\s*)?(?:json\s*)?(.*)(?:```)?", content, flags=re.DOTALL | re.IGNORECASE
            )  # must find something, at least return the entire text
            if not json_match:
                raise ValueError("Failed to find JSON in LLM response")
            json_str = json_match.group(1).strip()
            model_output = json.loads(json_str)
        else:
            # model_output = output.choices[0].message
            content = output.choices[0].message.content
            content = re.sub(r"```json\n|\n```", "", content)
            model_output = json.loads(content)
        return model_output

    def __call__(self, prompt: str, question: str):
        return self.forward(prompt, question)
