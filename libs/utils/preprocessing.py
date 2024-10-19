import torch
import torch.nn as nn


@torch.no_grad()
def multi_label_multi_class_one_hot_encode(labels, num_phases):
    """
        Convert a list of labels to a binary one hot encoding
        This function is for one hot encoding multi-label multi-class labels
    """
    labels = torch.LongTensor(labels)

    y_onehot = nn.functional.one_hot(labels, num_classes=num_phases)

    y_onehot = y_onehot.sum(dim=0).float()

    return y_onehot

@torch.no_grad()
def preprocessing(video_list,
                  feat_extractor_type,
                  max_seq_len, num_phases,
                  padding_val=0.0
                  ):
    """
        Generate batched features and masks from a list of dict items
    """
    if feat_extractor_type is not None:
        #each item in batch has frames as a list of tensors of shape Nwin x T x C x H x W
        frames = [x['window_frames'] for x in video_list]

        #B, Nwin x T x C x H x W -> B x Nwin x T x C x H x W
        batched_inputs = torch.stack(frames).cuda()

        batched_masks = torch.ones((len(video_list), 1, frames[0].shape[0])).cuda()

    else:
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()


        assert max_len <= max_seq_len, f"Input length must be smaller than max_seq_len during training, max len = {max_len}, max_seq_len = {max_seq_len}"
        # set max_len to self.max_seq_len
        max_len = max_seq_len
        # batch input shape B, C, T
        batch_shape = [len(feats), feats[0].shape[0], max_len]
        batched_inputs = feats[0].new_full(batch_shape, padding_val)
        for feat, pad_feat in zip(feats, batched_inputs):
            pad_feat[..., :feat.shape[-1]].copy_(feat)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

    actions = [multi_label_multi_class_one_hot_encode(x['actions_present'], num_phases) for x in video_list]

    gt_actions = torch.stack(actions, dim=0).cuda()

    gt_scores = torch.tensor([x['gt_score'] for x in video_list]).cuda()

    video_ids = [x['video_id'] for x in video_list]


    if 'target' in video_list[0]:
        target = torch.tensor([x['target'] for x in video_list]).cuda()
    else:
        target = []

    if "difficulty" in video_list[0]:
        difficulties = torch.tensor([x['difficulty'] for x in video_list])
    else:
        difficulties = []

    ret_dict = {
        'video_ids': video_ids,
        'batched_inputs': batched_inputs,
        'batched_masks': batched_masks,
        'gt_actions': gt_actions,
        'gt_scores': gt_scores,
        'target': target,
        'difficulties': difficulties
    }

    return ret_dict


