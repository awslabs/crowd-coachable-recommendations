import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DefaultDataCollator, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

# what's a better way to import these modules
from .vae_models import VAEPretrainedModel


def VAE_training(
    item_df,
    training_args=None,
    train_set_ratio=0.95,
    model_checkpoint="distilbert-base-uncased",
    max_length=350,
    vae_beta=2e-3,
    batch_size=64,
    callbacks=None,
):
    def tokenize_function(examples):
        result = tokenizer(
            examples["TITLE"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return result

    # data pre-process
    df = item_df.sample(frac=1, random_state=1).reset_index()
    df = df.drop(columns=["index"])

    size = len(df.index)
    df_train = df.iloc[: int(train_set_ratio * size)]
    df_test = df.iloc[int(train_set_ratio * size) :]
    tds = Dataset.from_pandas(df_train)
    # tds = tds.class_encode_column("label")

    vds = Dataset.from_pandas(df_test)
    # vds = vds.class_encode_column("label")

    ds = DatasetDict()
    ds["train"] = tds
    ds["test"] = vds

    # Tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = ds.map(
        tokenize_function, batched=True, remove_columns=["TITLE"]
    )

    # data_collator = DefaultDataCollator()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    logging_steps = len(tokenized_datasets["train"]) // batch_size

    # load pre-trained model
    model = VAEPretrainedModel.from_pretrained(model_checkpoint)
    model.VAE_post_init()
    model.set_beta(beta=vae_beta)
    model_name = "msmarco_VAE_model_prime_beta_" + str(vae_beta)

    if "training_args" not in dir():
        training_args = TrainingArguments(
            num_train_epochs=60,
            output_dir=f"{model_name}",
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            push_to_hub=False,
            fp16=True,
            logging_steps=logging_steps,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()
