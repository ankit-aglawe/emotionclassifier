# emotionclassifier/fine_tune.py
# Author: Ankit Aglawe

from transformers import Trainer, TrainingArguments

from emotionclassifier.logger import get_logger

logger = get_logger(__name__)


def fine_tune_model(model, tokenizer, train_dataset, val_dataset, output_dir, epochs=3):
    """Fine-tune a model with the given datasets and training arguments.

    Args:
        model: The model to fine-tune.
        tokenizer: The tokenizer for the model.
        train_dataset: The dataset for training.
        val_dataset: The dataset for validation.
        output_dir (str): The directory to save the model.
        epochs (int): The number of training epochs.
    """
    try:
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model fine-tuned and saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error in fine-tuning model: {e}")
        raise
