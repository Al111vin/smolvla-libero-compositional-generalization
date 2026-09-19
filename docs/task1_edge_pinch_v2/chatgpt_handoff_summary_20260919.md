# Task1 edge-pinch v2 — 交接摘要（供切换到 ChatGPT 继续执行使用）
日期：2026-09-19

本摘要目的：避免重复已完成实验，避免重犯已踩过的方法论坑，明确当前真实状态和禁止自行执行的操作边界。不改变任何官方结论或评测协议。

---

## 1. 当前结论

- **Task1 严格闭环评测：未通过。** 唯一可信证据：严格确定性协议下 batch1（555101–555105）× 6 checkpoint（500/1000/1500/2000/2500/3000）= 30 组合，结果 **0/30**（Wilson 95% CI [0.0, 0.1135]）。
- 旧协议（未固定 torch/cudnn 随机状态）下的 5/25(20%)@ckpt2500、3/25(12%)@ckpt3000 等数字仅作历史记录，**不再作为判定依据**。
- **Fold 02 继续锁定**，用户明确决定，非自动判定。
- `n_action_steps` 扫描和 `chunk_size` 缩小两个方向均**未获得支持**，不建议继续沿这两条线做参数扫描。
- 当前最值得优先深挖的方向：**555207（chunk10/ckpt3000）的失败机制**——它的释放动作本身是四条已捕获记录里最清晰、最稳健的（连续39帧超阈值开度，峰值远高于同批其他记录），但仍然失败，提示问题更可能出在**位置/姿态目标**，而不是释放时机本身。这个方向已有原始数据支持，不需要新 rollout。

---

## 2. 已验证证据（附具体文档出处）

- batch1 严格协议 0/30 → `task1_edge_pinch_v2_blind_eval_strict_determinism_v1_full_run_result_20260918.json`
- **3000:555207（baseline chunk50 checkpoint）**在严格协议复现性实验中稳定 **5/5 成功**，是目前唯一稳定复现的正成功案例 → `task1_edge_pinch_v2_official_conclusion_and_fold02_decision_20260917.json`
- **555207 用 chunk10 checkpoint 评测是失败案例**（跟上面是不同实验，同一个 seed 不同 checkpoint/group） → `chunk10_causal_experiment_result_20260918.json`, `release_action_capture_v1_result_20260919.json`
- 555207(chunk10) 释放动作稳健（连续39帧超阈值，峰值0.0785）但仍失败；555101/555102(baseline) 释放动作只是勉强达标（5–6帧，峰值0.018–0.022） → `release_step_robustness_audit_v1_result_20260919.json`
- 555102 释放窗口内夹爪命令变号次数（12次）和位置/旋转抖动是四条记录里最高的——这是它相对 555101 的独特信号（不是"释放未持续"） → 同上
- **gripper 符号约定**：action 值越大越闭合（open为负、close为正），两条独立证据方向一致 → `demo_release_action_profile_v1_result_20260919.json`, `gripper_timing_audit_v1_result_20260918.json`
- action 通道语义：`action[0:3]`=位置delta xyz，`action[3:6]`=旋转delta，`action[6]`=夹爪
- checkpoint 的 postprocessor 是纯线性反归一化（`raw=z*std+mean`），无 clip/tanh，捕获到的模型输出值（如555102的-0.183）是真实模型输出，不是处理artifact
- release_reopen_step 在三条失败记录里，k=5稳健检测下均不是单帧噪声误判 → `release_step_robustness_audit_v1_result_20260919.json`

---

## 3. 已排除或未获支持的假设（禁止重复实验）

| 假设 | 状态 | 依据 |
|---|---|---|
| 推理时 n_action_steps(1/5/10/25/50) 存在某值带来跨seed一致改善 | **未获支持** | 4seed×5值=20组合，无值在>1/4 seed成功，无seed在>2/5值成功，全是样本量为1的孤立观测（`n_action_steps_sweep_v1_result_20260918.json`） |
| 训练时缩小 chunk_size(50→10) 能改善释放 | **不支持，方向相反** | 受控实验：baseline@n10 是1/4成功；chunk10模型@n10 是0/4成功，且横向定位误差均值更差（`chunk10_causal_experiment_result_20260918.json`） |
| 模型从未发出明确张开指令 | **已排除** | 18/18条既有诊断轨迹均检测到release_reopen_step非None |
| HDF5→LeRobot转换破坏了夹爪数据 | **已排除** | 96/96条demo逐帧cross-check，max_abs_diff=0.0 |

---

## 4. 方法论陷阱（必读，否则会重复踩坑）

### (1) action_log 与 per_step_log 存在一帧偏移
`per_step_log` 含 reset 帧，长度=steps_run+1，step字段=自身下标。`action_log`/`quat_log` 只在每次 policy 调用后 append，长度=steps_run，0-indexed。**正确映射：`action_log_index = per_step_log_step - 1`**（step=0无对应action）。直接用同一 index 比较两个数组会系统性错位一帧。

### (2) 同一 seed 不保证逐帧轨迹一致；复现性判定应看 success/reward，不是逐帧对比
多处证据：
- diagnostic_rerun_v4 中 8 个此前 success=true 的组合重跑后只有 1 个（3000:555207）仍为 true，7 个翻转为 false
- chunk10 实验中 555207 在 smoke-test 单独运行 vs full-run 内运行，release_reopen_step 从 181 变为 107（配置逐字段一致）
- 本 session 中 555204 同配置 smoke-test vs full-run 的 release_reopen_step 也有 97 vs 99 的差异（success本身一致）

**含义**：即使在严格确定性协议下，仍存在未根治的残余不确定性（未 root-caused）。"多个独立 seed 可复现"这条 Fold02 解锁条件应理解为 **success/reward 层面的复现**，不能要求逐帧轨迹完全一致；任何"仅1次观测"的成功/失败案例不能当作稳定证据。

### (3) gripper 符号约定
open = 负值（约-1.0），close = 正值（约+1.0）。方向理解反了会把"释放"误判为"抓取加强"。

---

## 5. 禁止重复的已完成实验 / 需要单独确认的操作

**禁止重复：**
- n_action_steps sweep（已完成，无一致改善）
- chunk_size=10 训练+对照评测（已完成，方向相反）
- release_reopen_step 单帧误判核对（k=5，已完成4条记录）
- 555102 的 -0.183 是否processing artifact 的核查（已完成，是真实输出）
- HDF5 vs LeRobot 数据转换正确性核查（已完成，一致）

**新checkpoint/新训练输出目录规则：** 旧checkpoint（finetune_v1系列）与任何新训练产生的checkpoint必须用不同 output_dir，不得覆盖或混用现有结果目录。若"新checkpoint"其实是同一批旧checkpoint（如1500/2500）重新用严格协议评测，不需要重跑，直接引用 batch1 现有结果即可（已是0/30的一部分）。

**需要用户单独明确确认才能执行：**
- 解锁 Fold 02
- 删除任何文件
- 重启 GPU 或中断正在运行的进程
- 修改/切换 Git 分支，创建提交、推送、合并 PR
- 上传数据到 GitHub/HuggingFace（即使是轻量证据，也需先列清单+目标仓库+提交内容，等待确认）
- 并行启动可能冲突的训练/评测任务

**待核实，不要当作已确认事实：**
- GitHub 分支 `libero36-source-xy-redesign` 的 commit `5d1edc8` —— 本次交接材料整理中未能在 Project 已有文档里找到直接引用来源，使用前请先用 `git log` 实际核验。

---

## 6. 下一步候选（供参考，非最终决定）

1. **推荐优先**：555207(chunk10) 失败机制深挖，聚焦位置/姿态目标误差而非释放时机。已知 release_horizontal_distance 在 chunk10 模型上均值 0.239m（vs baseline 成功案例的 0.038m），可直接用已有的4条完整记录（含bowl/plate诊断位姿）做定量核对，不需要新 rollout。
2. **第二优先**：视觉对齐审计——gripper_timing_audit_v1 发现 9/18 案例是 "released_but_misaligned"，尚未专门审计过，可能和位置误差是同一根因的不同表现。
3. **谨慎对待**：任何新训练改动必须先说明数据来源/训练步数/输出目录/预计时间/成功判据；用不同 output_dir；不混入旧数据集；完成后仍需走严格确定性协议评测，成功判据必须是 `env.check_success()`，不能用 loss 或动作相似度代替。
4. **暂缓（用户已决定）**：batch2_extended（120组合）完整严格协议评测。
