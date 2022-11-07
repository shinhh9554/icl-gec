class Preprocessor:
    def __init__(self, args, tokenizer, label_to_id):
        self.args = args
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples['error'],
            truncation=True,
            max_length=self.args.max_seq_length
        )

        labels = []
        for i, label in enumerate(examples['tag']):
            label_ids = [-100] + [self.label_to_id[tag] for tag in label.split()][:self.args.max_seq_length - 2] + [-100]
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs