import argparse
from dataclasses import dataclass, field
from typing import List

# import open_clip
# from open_clip import tokenizer
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


"""
Format in the annotation pkl file
{'Forward': 1,
 'Back': 2,
 'Reverse': 3,
 'Inward': 4,
 'Arm.Forward': 5,
 'Arm.Back': 6,
 'Arm.Reverse': 7,
 '1 Som.Pike': 12,
 '1.5 Soms.Pike': 13,
 '2 Soms.Pike': 14,
 '2.5 Soms.Pike': 15,
 '3 Soms.Pike': 16,
 '3.5 Soms.Pike': 17,
 '4.5 Soms.Pike': 19,
 '1.5 Soms.Tuck': 21,
 '2 Soms.Tuck': 22,
 '2.5 Soms.Tuck': 23,
 '3 Soms.Tuck': 24,
 '3.5 Soms.Tuck': 25,
 '4.5 Soms.Tuck': 27,
 '0.5 Twist': 29,
 '1 Twist': 30,
 '1.5 Twists': 31,
 '2 Twists': 32,
 '2.5 Twists': 33,
 '3 Twists': 34,
 '3.5 Twists': 35,
 'Entry': 36,
 '0.5 Som.Pike': 37}
"""


@dataclass
class DataClass:
    sub_action_descriptions: List[str] = field(
        default_factory=lambda: [
            "Forward: A diving technique involving a front-facing takeoff and entry.",
            "Back: A diving technique involving a back-facing takeoff and entry.",
            "Reverse: A diving technique involving a back-facing takeoff and entry while rotating forward.",
            "Inward: A diving technique involving a front-facing takeoff and entry while rotating backwards.",
            "Arm Forward: A diving technique involving a front-facing takeoff and entry with arms extended and hands meeting above the head.",
            "Arm Back: A diving technique involving a back-facing takeoff and entry with arms extended and hands meeting above the head.",
            "Arm Reverse: A diving technique involving a back-facing takeoff and entry with arms extended and hands meeting above the head while rotating forward.",
            "0.5 Somersault Pike: A diving technique involving a take-off with half a somersault in the pike position before entering the water.",
            "1 Somersault Pike: A diving technique involving a takeoff and rotating forward to form a pike position with one somersault.",
            "1.5 Somersaults Pike: A diving technique involving a takeoff and rotating forward to form a pike position with one and a half somersaults.",
            "2 Somersaults Pike: A diving technique involving a takeoff and rotating forward to form a pike position with two somersaults.",
            "2.5 Somersaults Pike: A diving technique involving a takeoff and rotating forward to form a pike position with two and a half somersaults.",
            "3 Somersaults Pike: A diving technique involving a takeoff and rotating forward to form a pike position with three somersaults.",
            "3.5 Somersaults Pike: A diving technique involving a takeoff and rotating forward to form a pike position with three and a half somersaults.",
            "4.5 Somersaults Pike: A diving technique involving a takeoff and rotating forward to form a pike position with four and a half somersaults.",
            "1.5 Somersaults Tuck: A diving technique involving a takeoff and rotating forward to bend at the waist with one and a half somersaults.",
            "2 Somersaults Tuck: A diving technique involving a takeoff and rotating forward to bend at the waist with two somersaults.",
            "2.5 Somersaults Tuck: A diving technique involving a takeoff and rotating forward to bend at the waist with two and a half somersaults.",
            "3 Somersaults Tuck: A diving technique involving a takeoff and rotating forward to bend at the waist with three somersaults.",
            "3.5 Somersaults Tuck: A diving technique involving a takeoff and rotating forward to bend at the waist with three and a half somersaults.",
            "4.5 Somersaults Tuck: A diving technique involving a takeoff and rotating forward to bend at the waist with four and a half somersaults.",
            "0.5 Twist: A diving technique involving a takeoff and half a twist before entering the water.",
            "1 Twist: A diving technique involving a takeoff and one full twist before entering the water.",
            "1.5 Twists: A diving technique involving a takeoff and one and a half twists before entering the water.",
            "2 Twists: A diving technique involving a takeoff and two full twists before entering the water.",
            "2.5 Twists: A diving technique involving a takeoff and two and a half twists before entering the water.",
            "3 Twists: A diving technique involving a takeoff with three twists before entering the water.",
            "3.5 Twists: A diving technique involving a takeoff with three and a half twists before entering the water.",
            "Entry: A diving technique involving a entry into the water, typically performed at the end of a dive."
        ]
    )

class GetTextEmbeddings:
    def __init__(self, output_path) -> None:
        """
        Args:
            output_path (_type_): output_path to save the embeddibngs 
        """
        self.data = DataClass()
        self.output_path = output_path

    def get_huggingface_embeddings(self):
        def average_pool(
            last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            last_hidden = last_hidden_states.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        input_texts = [
            "This is a " + desc.replace(": ", " action: ")
            for desc in self.data.sub_action_descriptions
        ]

        model_id="google/flan-t5-xxl" # "intfloat/e5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        

        # Tokenize the input texts
        batch_dict = tokenizer(
            input_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs= model(input_ids= batch_dict['input_ids'], decoder_input_ids=batch_dict['input_ids'])
            embeddings = average_pool(
                outputs.last_hidden_state, batch_dict["attention_mask"]
            )
        np.save(self.output_path, embeddings.detach().cpu().numpy())
        return embeddings.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        help="path to save the embeddings",
        nargs="?"
    )
    args = parser.parse_args()

    get_text_embeddings = GetTextEmbeddings("FineDiving_t5_xxl_text_embeddings.npy")
    text_embeddings = get_text_embeddings.get_huggingface_embeddings()
    print("Text Embed shape: ", text_embeddings.shape)

