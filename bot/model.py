import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from config import USE_GPU


class QuestionAnsweringModel:
    def __init__(self, model_name: str, state_dict_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

        device = torch.device(
            "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
        )
        self.model.load_state_dict(
            torch.load(state_dict_path, map_location=device)
        )
        self.model.to(device)
        self.model.eval()

    def get_answer(self, question: str) -> str:
        inputs = self.tokenizer(
            question,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits

        answer_start = int(torch.argmax(start_scores))
        answer_end = int(torch.argmax(end_scores))

        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end]
        )
        return answer
