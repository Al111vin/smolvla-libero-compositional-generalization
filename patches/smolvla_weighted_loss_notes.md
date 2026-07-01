# SmolVLA Weighted Loss Notes

During V1.5 and V1.6 task0 overfit experiments, the SmolVLA internal flow-matching loss was temporarily modified in:

lerobot/policies/smolvla/modeling_smolvla.py

Modified after this line:
losses = losses[:, :, :original_action_dim]

V1.5 gripper-weighted loss:
action_loss_weights = torch.ones_like(losses)
action_loss_weights[..., 6] = 5.0
losses = losses * action_loss_weights

This improved action_6 / gripper MAE.

V1.6 motion + gripper weighted loss:
action_loss_weights = torch.ones_like(losses)
action_loss_weights[..., 0] = 2.0
action_loss_weights[..., 1] = 2.0
action_loss_weights[..., 2] = 3.0
action_loss_weights[..., 6] = 5.0
losses = losses * action_loss_weights

V1.6 did not improve overall dataset MAE compared with V1.5.
The site-packages source file was restored after the experiments.
