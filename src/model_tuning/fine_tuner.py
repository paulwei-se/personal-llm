from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import torch
import asyncio

# this is not ready yet
class ModelFineTuner:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self, examples):
        """Prepare data for fine-tuning."""
        inputs = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    async def fine_tune(self, document, questions_answers, output_dir, num_train_epochs=3):
        train_data = {
            "context": [document] * len(questions_answers),
            "question": [qa["question"] for qa in questions_answers],
            "answers": [{"answer_start": [document.index(qa["answer"])], "text": [qa["answer"]]} for qa in questions_answers]
        }
        eval_data = {k: v[:1] for k, v in train_data.items()}

        train_dataset = Dataset.from_dict(train_data)
        eval_dataset = Dataset.from_dict(eval_data)

        train_dataset = train_dataset.map(self.prepare_data, batched=True, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(self.prepare_data, batched=True, remove_columns=eval_dataset.column_names)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        await asyncio.to_thread(trainer.train)
        await asyncio.to_thread(trainer.save_model, output_dir)
        await asyncio.to_thread(self.tokenizer.save_pretrained, output_dir)

# Usage example
if __name__ == "__main__":
    fine_tuner = ModelFineTuner()

    # Sample data (you would typically load this from a file)
    train_data = {
        "question": ["What is the capital of France?", "Who wrote Hamlet?"],
        "context": ["Paris is the capital of France.", "Hamlet was written by William Shakespeare."],
        "answers": [
            {"answer_start": [0], "text": ["Paris"]},
            {"answer_start": [22], "text": ["William Shakespeare"]}
        ]
    }
    eval_data = {
        "question": ["What is the largest planet in our solar system?"],
        "context": ["Jupiter is the largest planet in our solar system."],
        "answers": [
            {"answer_start": [0], "text": ["Jupiter"]}
        ]
    }

    fine_tuner.fine_tune(train_data, eval_data, "fine_tuned_model")
    print("Fine-tuning complete. Model saved in 'fine_tuned_model' directory.")
