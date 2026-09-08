using NUnit.Framework;
using UnityEngine;

/// <summary>
/// K1/G2a: 补偿函数的映射与符号测试（EditMode，纯函数）
///
/// 验证：
/// 1. 任一时刻只有一个膝关节非零
/// 2. 非零的那个是支撑腿的膝（uf1高→右腿支撑→右膝；反之左膝）
/// 3. rollDeg 反号时补偿反号
/// </summary>
public class G2a_CompensationMappingTests
{
    [Test]
    public void TestCompensationMapping()
    {
        // 创建测试用的 GameObject 和组件
        GameObject go = new GameObject("TestRobot");
        RobotRLAgent agent = go.AddComponent<RobotRLAgent>();
        PredictiveFeedforwardController ffController = go.AddComponent<PredictiveFeedforwardController>();

        // K1: 初始化关节索引映射（模拟 D1）
        agent.IdxLeftKnee = 3;
        agent.IdxRightKnee = 9;
        agent.ActionNum = 12;

        // K1: 手动初始化 controller（EditMode 不会自动调用 Start）
        ffController.Initialize();

        // K1: 重置滤波器
        ffController.ResetFilter();

        // K1: 测试参数（只测 ±5°，不测 0° 避免滤波器瞬态问题）
        float[] rollDegrees = { +5f, -5f };
        (float uf1, float uf2, string supportLeg, int expectedKneeIdx)[] phaseConfigs =
        {
            (0.9f, 0.0f, "右腿支撑", agent.IdxRightKnee),  // uf1 高 → 左腿摆动 → 右腿支撑
            (0.0f, 0.9f, "左腿支撑", agent.IdxLeftKnee)   // uf2 高 → 右腿摆动 → 左腿支撑
        };

        Debug.Log("[K1/G2a] === 补偿函数映射与符号测试 ===");

        foreach (var config in phaseConfigs)
        {
            Debug.Log($"[K1] --- {config.supportLeg} (uf1={config.uf1}, uf2={config.uf2}) ---");

            // K1: 测试每个 roll 值（重置滤波器以避免历史影响）
            foreach (float rollDeg in rollDegrees)
            {
                // K1: 重置滤波器并连续调用让其收敛到当前 roll 值
                ffController.ResetFilter();
                for (int i = 0; i < 10; i++)
                {
                    ffController.ComputeKneeCompensation(new Vector3(0f, 0f, rollDeg), config.uf1, config.uf2, agent.ActionNum);
                }

                Vector3 predictedEuler = new Vector3(0f, 0f, rollDeg);

                // K1: 调用补偿函数
                float[] compensation = ffController.ComputeKneeCompensation(predictedEuler, config.uf1, config.uf2, agent.ActionNum);

                // K1: 统计非零关节
                int nonZeroCount = 0;
                int nonZeroIdx = -1;
                float nonZeroValue = 0f;

                for (int i = 0; i < agent.ActionNum; i++)
                {
                    if (Mathf.Abs(compensation[i]) > 0.001f)
                    {
                        nonZeroCount++;
                        nonZeroIdx = i;
                        nonZeroValue = compensation[i];
                    }
                }

                // K1: 断言1 - 任一时刻只有一个膝关节非零
                Assert.AreEqual(1, nonZeroCount,
                    $"roll={rollDeg}°, {config.supportLeg}: 应只有1个膝关节非零，实际有 {nonZeroCount} 个");

                // K1: 断言2 - 非零的那个是支撑腿的膝
                Assert.AreEqual(config.expectedKneeIdx, nonZeroIdx,
                    $"roll={rollDeg}°, {config.supportLeg}: 非零关节应是索引 {config.expectedKneeIdx}（{GetJointName(config.expectedKneeIdx, agent)}），实际是 {nonZeroIdx}");

                Debug.Log($"[K1] roll={rollDeg:+0;-0}° → 索引[{nonZeroIdx}]({GetJointName(nonZeroIdx, agent)}) = {nonZeroValue:F4}°");
            }

            // K1: 断言3 - rollDeg 反号时补偿反号
            // 重新计算 +5° 和 -5° 的补偿（确保滤波器稳定）
            for (int i = 0; i < 5; i++)
            {
                ffController.ComputeKneeCompensation(new Vector3(0, 0, +5f), config.uf1, config.uf2, agent.ActionNum);
            }
            float[] comp_pos5 = ffController.ComputeKneeCompensation(new Vector3(0, 0, +5f), config.uf1, config.uf2, agent.ActionNum);

            ffController.ResetFilter();  // 重置避免历史影响
            for (int i = 0; i < 5; i++)
            {
                ffController.ComputeKneeCompensation(new Vector3(0, 0, -5f), config.uf1, config.uf2, agent.ActionNum);
            }
            float[] comp_neg5 = ffController.ComputeKneeCompensation(new Vector3(0, 0, -5f), config.uf1, config.uf2, agent.ActionNum);

            float val_pos5 = comp_pos5[config.expectedKneeIdx];
            float val_neg5 = comp_neg5[config.expectedKneeIdx];

            Assert.IsTrue(val_pos5 * val_neg5 < 0,
                $"{config.supportLeg}: roll 反号时补偿应反号，+5°={val_pos5:F4}, -5°={val_neg5:F4}");

            Debug.Log($"[K1] ✓ 反号验证: +5°={val_pos5:F4}°, -5°={val_neg5:F4}°");
        }

        Debug.Log("[K1/G2a] === 所有断言通过 ===");

        // 清理
        Object.DestroyImmediate(go);
    }

    string GetJointName(int idx, RobotRLAgent agent)
    {
        if (idx == agent.IdxLeftKnee) return "左膝";
        if (idx == agent.IdxRightKnee) return "右膝";
        return $"索引{idx}";
    }
}
