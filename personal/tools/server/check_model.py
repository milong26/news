# 验证方法
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# 1. 查看所有参数
for name, param in policy.named_parameters():
    print(f"{name}: {param.shape}, 可训练: {param.requires_grad}")

# 2. 按模块统计参数量
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

print(f"VLM+Expert总参数: {count_parameters(policy.model.vlm_with_expert):,}")
print(f"State投影: {count_parameters(policy.model.state_proj):,}")
print(f"Action输入投影: {count_parameters(policy.model.action_in_proj):,}")
print(f"Action输出投影: {count_parameters(policy.model.action_out_proj):,}")
print(f"Action时间MLP: {count_parameters(policy.model.action_time_mlp_in) + count_parameters(policy.model.action_time_mlp_out):,}")

# 3. 查看哪些参数被冻结
vlm_params = count_parameters(policy.model.vlm_with_expert)
trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"\n总参数: {count_parameters(policy):,}")
print(f"可训练参数: {trainable_params:,}")