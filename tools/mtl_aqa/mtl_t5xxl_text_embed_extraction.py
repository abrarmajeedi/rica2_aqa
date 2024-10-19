from dataclasses import dataclass, field
from typing import List

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

##########################MTL########
@dataclass
class DataClass:
    sub_action_descriptions: List[str] = field(
        default_factory=lambda: [
            "Armstand: In this position, athletes start by standing on their hands on the edge of the diving board and perform their dive while maintaining this handstand position.",
            "Inwards: In this rotation type, athletes perform a forward-facing takeoff and rotate inward toward the diving board as they execute their dive.",
            "Reverse: In this rotation type, athletes perform a backward-facing takeoff and rotate backward away from the diving board as they execute their dive.",
            "Backward: In this rotation type, athletes perform a backward-facing takeoff and rotate backward toward the diving board as they execute their dive.",
            "Forward: In this rotation type, athletes perform a forward-facing takeoff and rotate forward away from the diving board as they execute their dive.",
            "Free: In this position, athletes have the freedom to perform any combination of dives from various categories without any restrictions or limitations.",
            "Tuck: In this position, athletes bring their knees to their chest and hold onto their shins while maintaining a compact shape throughout their dive.",
            "Pike: In this position, athletes maintain a straight body with their legs extended and their toes pointed out while bending at the waist to bring their hands toward their toes.",
            "0.5 Somersault: Athletes perform a half rotation in the air during their dive.",
            "1 Somersault: Athletes perform a full forward or backward rotation in the air during their dive.",
            "1.5 Somersault: Athletes perform a full rotation and an additional half rotation in the air during their dive.",
            "2 Somersault: Athletes perform two full forward or backward rotations in the air during their dive.",
            "2.5 Somersault: Athletes perform two full rotations and an additional half rotation in the air during their dive.",
            "3 Somersault: Athletes perform three full forward or backward rotations in the air during their dive.",
            "3.5 Somersault: Athletes perform three full rotations and an additional half rotation in the air during their dive.",
            "4.5 Somersault: Athletes perform four full rotations and an additional half rotation in the air during their dive.",
            "0.5 Twist: Athletes perform a half twist in the air during their dive.",
            "1 Twist: Athletes perform one full twist in the air during their dive.",
            "1.5 Twist: Athletes perform one and a half twists in the air during their dive.",
            "2 Twist: Athletes perform two full twists in the air during their dive.",
            "2.5 Twist: Athletes perform two and a half twists in the air during their dive.",
            "3 Twist: Athletes perform three full twists in the air during their dive.",
            "3.5 Twist: Athletes perform three and a half twists in the air during their dive.",
            "Entry: A diving technique involving a entry into the water, typically performed at the end of a dive.",
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


get_text_embeddings = GetTextEmbeddings("MTL_t5_xxl_text_embeddings.npy")
text_embeddings = get_text_embeddings.get_huggingface_embeddings()
print("Text Embed shape: ", text_embeddings.shape)
