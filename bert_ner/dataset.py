import torch


class Dataset:
    CLS_TOK = 101
    SEP_TOK = 102
    NULL_TOK = 0
    def __init__(self, texts, pos, tags, tokenizer, max_len):
        self.texts = texts
        self.pos = pos
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids, target_pos, target_tag = [], [], []
        for i, s in enumerate(text):
            inputs = self.tokenizer.encode(
                s, add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]] * input_len)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:self.max_len - 2]
        target_pos = target_pos[:self.max_len - 2]
        target_tag = target_tag[:self.max_len - 2]

        ids = [self.CLS_TOK] + ids + [self.SEP_TOK]
        target_pos = [self.NULL_TOK] + target_pos + [self.NULL_TOK]
        target_tag = [self.NULL_TOK] + target_tag + [self.NULL_TOK]

        mask = [1] * len(ids)
        token_type_ids = [self.NULL_TOK] * len(ids)

        padding_len = self.max_len - len(ids)

        ids = ids + ([self.NULL_TOK] * padding_len)
        mask = mask + ([self.NULL_TOK] * padding_len)
        token_type_ids = token_type_ids + ([self.NULL_TOK] * padding_len)
        target_pos = target_pos + ([self.NULL_TOK] * padding_len)
        target_tag = target_tag + ([self.NULL_TOK] * padding_len)

        return dict(
            ids=torch.tensor(ids, dtype=torch.long),
            mask=torch.tensor(mask, dtype=torch.long),
            token_type_ids=torch.tensor(token_type_ids, dtype=torch.long),
            target_pos=torch.tensor(target_pos, dtype=torch.long),
            target_tag=torch.tensor(target_tag, dtype=torch.long),
        )