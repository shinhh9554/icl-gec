class Preprocessor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def preprocess_function(self, examples):
        inputs = examples["error"]
        targets = examples["correct"]

        # Tokenize inputs/targets
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_input_length, truncation=True)
        labels = self.tokenizer(targets, max_length=self.args.max_target_length - 1, truncation=True)

        # add eos_token_id
        model_inputs["labels"] = [target_ids + [self.tokenizer.eos_token_id] for target_ids in labels["input_ids"]]
        return model_inputs
