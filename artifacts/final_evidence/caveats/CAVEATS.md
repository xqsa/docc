# ARAC-lite Final Caveats

- 日期：2026-05-21
- 执行者：Codex

## 结论边界

- V0.7 支持 V0.6-targeted-probe 作为稳定候选：6/6 问题相对 disable-fast 为 majority non-worse，额外 FE 低于 1%。
- V0.8 没有证明 targeted probe 明显优于 same-budget random probe；targeted selection 不能作为主贡献。
- V0.8 delta gate stress test 显示 accept-only recovery 会显著放大 R6 bad recovery；delta gate 是当前最强机制证据。
- 当前版本定位为 ARAC-lite，不是完整 ARAC、UCB 或 bandit 版本。
