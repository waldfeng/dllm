# 专利 κ·δ/τ_s 选择项实施计划（v3：脚本统一专利默认）

分支：`patent-kappa-selection`（已从 50a2988eb12e5f6cded46b3b074fc7cdde389966 切出）

## 设计（用户已拍板）

1. `--loss-weight-prior` **只收 5 值**（和为 1，默认 "0.65,0.06,0.06,0.17,0.06" = 专利示例）；6 值输入报错并给迁移提示
2. κ 独立参数 `--exit-selection-delta-weight`，默认 **0.05**（专利示例）；τ_s 复用 `--exit-tau-delta`=0.05
3. **设计 B（专利字面）**：`d_i = argmin(ℓ + κ·δ/τ_s)`，κ·δ/τ_s 只进 argmin；loss_A 的 A1/ρ/熵保持五项 ℓ
4. 旧六项逐位等价由测试内直接构造权重证明（不经 CLI）
5. **脚本统一专利默认**：smoke 与旧 launch_joint.sh 删掉 6 值 prior 覆盖，全部用 CLI 默认

## 本轮核查修正的事实（2026-10-26）

- **真实 1-epoch launcher（temp/launch_joint_1epoch.sh）本就无 prior/κ 覆盖** → 一直用专利默认点。此前审计把 6 值 CE=0 配置归到它是错误归属（只在旧 launch_joint.sh 与 smoke 中）→ 审计发现 #3 降级为"仅辅助脚本偏离"。真实训练零改动。
- **50a2988 的 joint 保存块（3 处）不含 exit_loss_weights/exit_temperature**（只有 hard/guard min depth、entropy beta、objective v3）→ 本计划"保存块补三键"是修复既有缺口。
- **smoke_exit_e2e.py 在 50a2988 上从未跑通**（跨版本陈旧）：
  - L142 `--gradient-accumulation 2` 与 joint 模式强制 =1 冲突 → 必失败
  - L154 断言 `exit_loss_weights/exit_temperature in payload` → 保存块缺键，必失败
  - L155 断言 `loss_weights_raw in payload` → 50a2988 从不保存该键（更早的可训练权重时代），必失败
  - L159/160 断言 CE<0.05、speed≥0.3 → 旧实验配置的硬编码契约
  - 本次实施把 smoke 整体拉回可用状态（断言按新契约重写）

## 代码改动

### train.py

**1. `parse_args`**
- 新增 `--exit-selection-delta-weight`（float，默认 0.05，help 说明专利规则与 τ_s 来源）
- `--loss-weight-prior` help 重写为五值语义 + 六值退役提示

**2. prior 校验块（train_joint_exit ~1170）**
```python
    prior = torch.tensor(
        [float(value) for value in args.loss_weight_prior.split(",")], dtype=torch.float32
    )
    if prior.numel() != 5 or abs(float(prior.sum()) - 1.0) > 1e-4:
        raise ValueError(
            "--loss-weight-prior must hold exactly five values summing to one: "
            "[CE, confidence, margin, speed, guard]; the legacy delta slot moved "
            "to --exit-selection-delta-weight"
        )
    kappa = float(args.exit_selection_delta_weight)
    if kappa < 0:
        raise ValueError("--exit-selection-delta-weight must be non-negative")
```

**3. `joint_sample_losses`（~1123）**——设计 B 核心
- 签名加 `delta_costs=None, kappa=0.0`
- `selection_costs = costs + kappa * delta_costs`（带 shape 校验）
- `chosen = selection_costs[floor:].detach().argmin(0) + floor`（唯一改动的既有行）
- A1 的 `costs[chosen, sample]`、ρ/熵的 `candidate = costs[floor:]` 保持五项

**4. Phase A（train_joint_exit）**
- ~1339 `compute_deltas=False` → `True`
- ~1344 grab 块加 `deltas = rollout["deltas"].clone()`
- ~1354 调用传 `delta_costs=deltas / args.exit_tau_delta, kappa=kappa`

**5. resume 告警 + 打印**
- resume 载荷 `exit_selection_delta_weight` 与 CLI 不一致 → 打印告警（保留 CLI 值）
- `joint_start` 行追加 ` kappa={kappa}`；`joint_depth` 行/diagnostic dict 追加 `"selection_kappa": kappa`

**6. 3 处保存 payload 追加**
```python
                        "exit_loss_weights": fixed_weights.detach().cpu().tolist(),
                        "exit_temperature": args.exit_temp_end,
                        "exit_selection_delta_weight": kappa,
```
`joint_objective`: `"sample_a1_a2_v3"` → `"sample_a1_a2_v4"`

### memory_transformer.py

- `joint_rollout` docstring：δ 用途改为"供 phase A 选择项 κ·δ/τ_s（仅 argmin）"，删旧说法。本体零改动

### tests/test_early_exit.py（4 个新测试）

1. `test_selection_delta_term_matches_six_term_reference`：小模型 rollout，代码内构造不归一五值+κ = 旧六项规格，对照 `assemble_exit_terms(include_delta=True)` 六项加权 → chosen 全等 + 选择代价 allclose
2. `test_delta_costs_do_not_enter_loss_a`：均匀 δ 行 → A1/A2/ρ/熵与 κ=0 逐位相同
3. `test_delta_term_steers_selection_deeper`：合成 δ 场 → 选择变深；κ=0 不变
4. `test_joint_prior_validation`：5 值和=1 通过；6 值/和≠1/κ<0 报错

### temp/smoke_exit_e2e.py（拉回可用 + 专利默认）

- L142：`--gradient-accumulation 2` → `1`
- L145：删除 `--loss-weight-prior` 行（用默认 = 专利示例）
- L154：保留（保存块补键后成立）
- L155：删 `loss_weights_raw` 断言，保留 `optimizer_readout`
- L158：保留（和=1）
- L159/160 翻转：`weights[0] >= 0.6`（CE 主导）、`weights[3] <= 0.2`（speed 非主导；注意五值布局 speed 在 index 3）
- 后续 phase-isolation / resume 段按实际跑通情况微调

### temp/launch_joint.sh

- 删除 `--loss-weight-prior "0.0,..."` 行（用默认）；脚本本身已过时，注明被 launch_joint_1epoch.sh 取代

### temp/launch_joint_1epoch.sh / run_joint_1000.py

- **零改动**（已用默认 = 专利示例点）

## 初期动力学说明（无需额外机制）

readout 恒等初始化（B=c=0, s=1）下 CE≈lnV、conf≈1/V 各深度近似持平 → 选择由 speed(0.17)+guard+κ·δ/τ_s(系数 1.0) 主导，起步偏浅；loss_A 训练后 CE 分化，选择迁向预测最优深度。κ=0.05/τ_s=0.05 的 δ 系数 1.0 与旧默认六项 (0.05/0.05) 一致。

## 验证

1. `.venv/bin/python -m py_compile train.py memory_transformer.py`
2. `.venv/bin/python -m pytest tests/ -q`（GPU1 ~17GB 空闲）
3. `CUDA_VISIBLE_DEVICES=1 .venv/bin/python temp/smoke_exit_e2e.py`
4. 预期：κ>0 → 选择偏深、E[t] 上升；κ=0 → 与现行为一致；smoke 断言 weights=[0.65,0.06,0.06,0.17,0.06]

## 提交

单次提交到 `patent-kappa-selection`：

```
Patent selection rule: standalone kappa*delta/tau_s in the joint depth argmin

- joint_sample_losses takes delta_costs/kappa: the state-change row enters
  the selection cost only (d_i = argmin(l + kappa*delta/tau_s), patent
  literal); A1, rho and the entropy stay on the five-term cost
- --loss-weight-prior is five-value only; the legacy delta slot moved to
  the standalone --exit-selection-delta-weight (default 0.05, patent
  example); six-value input fails with a migration hint
- phase A rolls out with compute_deltas=True again; joint checkpoints save
  exit_loss_weights/exit_temperature/exit_selection_delta_weight (the
  weight/temperature keys were missing entirely) and warn on kappa mismatch
  at resume; objective tag sample_a1_a2_v4
- smoke_exit_e2e.py revived for the current trainer (grad-accum 1, default
  patent-example prior, assertion contract flipped to CE-dominant weights);
  launch scripts unified on the patent example point
- tests: six-term reference equivalence, loss_A exclusion, steering, and
  prior/kappa validation
```
