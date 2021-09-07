import torch
import transformers

def loss_fun(output, target, mask, num_labels):
    loss = torch.nn.CrossEntropyLoss()
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        mask.view(-1) == 1,
        target.view(-1),
        torch.tensor(loss.ignore_index).type_as(target)
    )
    loss_value = loss(active_logits, active_labels)
    return loss_value


class Model(torch.nn.Module):
    def __init__(self, model_path, n_tag, n_pos):
        super().__init__()
        
        self.n_tag = n_tag
        self.n_pos = n_pos
        self.model_path = model_path
        
        self.bert = transformers.BertModel.from_pretrained(
            self.model_path,return_dict=False
        )
        self.drop_tag = torch.nn.Dropout(0.3)
        self.drop_pos = torch.nn.Dropout(0.3)
        self.linear_tag = torch.nn.Linear(768, self.n_tag)
        self.linear_pos = torch.nn.Linear(768, self.n_pos)
    
    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        out_bert, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        output_tag = self.linear_tag(self.drop_tag(out_bert))
        output_pos = self.linear_pos(self.drop_pos(out_bert))

        loss_tag = loss_fun(output_tag, target_tag, mask, self.n_tag)
        loss_pos = loss_fun(output_pos, target_pos, mask, self.n_pos)

        loss = (loss_tag + loss_pos) / 2

        return output_tag, output_pos, loss
