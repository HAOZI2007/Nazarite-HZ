import torch
import torch.nn.functional as F


class PolicyValueTrainer:
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-4,
        value_weight=1.0,
    ):
        self.model = model
        self.value_weight = value_weight
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr= learning_rate,
            weight_decay= weight_decay,
        )

    def update(self,batch):
        states, target_policies, target_outcomes = batch

        target_policies = target_policies.detach()
        target_outcomes = target_outcomes.detach()

        self.model.train()

        self.optimizer.zero_grad(set_to_none= True)

        policy_logits, predicted_values = self.model(states)
        log_probs = F.log_softmax(policy_logits, dim=1)

        policy_loss = -(
            target_policies * log_probs
        ).sum(dim=1).mean()

        value_loss = F.mse_loss(
            predicted_values.squeeze(-1),
            target_outcomes,
        )

        loss = policy_loss + self.value_weight * value_loss

        if not torch.isfinite(loss).item():
            raise ValueError("损失出现 NaN 或无穷值")

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        self.optimizer.step()
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }