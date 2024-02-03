import json
import numpy as np
from typing import Optional
from datasets.arrow_dataset import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from .dataset import DataTrainingArguments, normalize, serialize_schema
from .trainer import Seq2SeqTrainer, EvalPrediction


def nl2formula_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()


def nl2formula_get_target(
    query: str,
    normalize_query: bool,
) -> str:
    _normalize = normalize if normalize_query else (lambda x: x)
    return _normalize(query)


def nl2formula_add_serialized_schema(ex: dict, data_training_args: DataTrainingArguments) -> dict:
    """
    serialized_schema = serialize_schema(
        question=ex["Question"],
        db_column_names=ex["db_column_names"],
        db_table_names=ex["db_table_names"],
        schema_serialization_type=data_training_args.schema_serialization_type,
        schema_serialization_randomized=data_training_args.schema_serialization_randomized,
        schema_serialization_with_db_id=data_training_args.schema_serialization_with_db_id,
        schema_serialization_with_db_content=data_training_args.schema_serialization_with_db_content,
        normalize_query=data_training_args.normalize_query,
    )
    """
    serialized_schema=" | "
    for row in ex["Table"]:
        for coloum in row:
            serialized_schema+=coloum+", "
    return {"serialized_schema": serialized_schema}


def nl2formula_pre_process_function(
    batch: dict,
    max_source_length: Optional[int],
    max_target_length: Optional[int],
    data_training_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    prefix = data_training_args.source_prefix if data_training_args.source_prefix is not None else ""

    inputs = [
        nl2formula_get_input(question=question, serialized_schema=serialized_schema, prefix=prefix)
        for question, serialized_schema in zip(batch["Question"], batch["serialized_schema"])
    ]

    model_inputs: dict = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=False,
        truncation=True,
        return_overflowing_tokens=False,
    )

    targets = [
        nl2formula_get_target(
            query=query,
            normalize_query=data_training_args.normalize_query,
        )
        for query in batch["Formula"]
    ]

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_overflowing_tokens=False,
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class NL2formulaTrainer(Seq2SeqTrainer):
    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        inputs = self.tokenizer.batch_decode([f["input_ids"] for f in features], skip_special_tokens=True)
        label_ids = [f["labels"] for f in features]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            _label_ids = np.where(label_ids != -100, label_ids, self.tokenizer.pad_token_id)
        decoded_label_ids = self.tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
        levels=[example["Level"] for example in examples]

        metas = [
            {
                "context": context,
                "label": label,
                "level":level,
            }
            for context, label, level in zip(inputs, decoded_label_ids,levels)
        ]
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        assert len(metas) == len(predictions)
        with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
            json.dump(
                [dict(**{"prediction": prediction}, **meta) for prediction, meta in zip(predictions, metas)],
                f,
                indent=4,
            )
        return EvalPrediction(predictions=predictions, label_ids=label_ids, metas=metas)

    def _compute_metrics(self, eval_prediction: EvalPrediction)->dict:
        predictions, label_ids, metas = eval_prediction
        """
        if self.target_with_db_id:
            # Remove database id from all predictions
            predictions = [pred.split("|", 1)[-1].strip() for pred in predictions]
        """
        # TODO: using the decoded reference labels causes a crash in the spider evaluator
        # if self.ignore_pad_token_for_loss:
        #     # Replace -100 in the labels as we can't decode them.
        #     label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
        # decoded_references = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # references = [{**{"query": r}, **m} for r, m in zip(decoded_references, metas)]
        references = metas
        return self.metric.compute(predictions=predictions, references=references)
