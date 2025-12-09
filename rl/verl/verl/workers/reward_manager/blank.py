# Khiem: Add blank reward manager for PURE-PRM training
# Adapt from PURE
from verl import DataProto
import torch


class BlankRewardManager:
    """The blank reward manager for PURE-PRM training.
    
    This manager returns zero verifiable rewards, allowing the model to train
    exclusively on process reward model (PRM) feedback without requiring
    ground truth answers.
    
    Use this for PURE-PRM variant where only process-level supervision is used.
    """

    def __init__(self, tokenizer, num_examine=1, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        # For PURE-PRM, we only use process rewards (no verifiable rewards)
        # Print a few examples for debugging if needed
        if self.num_examine > 0:
            prompt_ids = data.batch['prompts']
            prompt_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            
            response_ids = data.batch['responses']
            sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            rm_scores = data.batch.get('rm_scores', None)
            
            for i in range(min(self.num_examine, len(data))):
                output_str = f"{'Rollout Example ' + str(i+1):#^{50}}\n"
                output_str += f"Question:\n{repr(prompt_str[i])}\n"
                steps = sequences_str[i].split('\n\n')
                output_str += f"Rollout:\n{repr(sequences_str[i])}\n"
                output_str += f"Num of steps: {len(steps)}\n"
                if rm_scores is not None:
                    outcome_reward = rm_scores[i].sum()
                    output_str += f"Outcome reward: {outcome_reward.item()}\n"
                print(output_str.rstrip())
            
            # Only print once per batch
            self.num_examine = 0
        
        # Return zero verifiable rewards for all responses
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        output = DataProto.from_dict({
            "verifiable_rewards": reward_tensor.sum(-1),
            "reward_fn_scores": reward_tensor,
        })

        return output
