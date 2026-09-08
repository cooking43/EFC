using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;
using NUnit.Framework;

#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// P0-1 自动化验证测试（重写版）
///
/// 验证真实场景中真实机器人的真实实现：
/// 1. 整机质量 ≈ 35kg（Unitree G1 官方规格）
/// 2. 质心 x/z 随时间变化（非恒定）
/// 3. 单 link 质心 vs 全身质心有差异
///
/// 验收标准：产出 XML 测试结果和控制台日志
/// </summary>
public class CoMVerificationTests
{
    // Unitree G1 23-dof官方标称质量：35kg（含9000mAh电池，约2-3kg）
    // 来源：https://humanoid.guide/product/g1/ & https://robostore.com/
    // Unity场景测得：32.454kg（未建模电池）→ 偏差7.3%合理
    private const float UNITREE_G1_NOMINAL_MASS = 35.0f;
    private const float MASS_TOLERANCE_PERCENT = 10.0f;
    private const int TEST_FIXEDUPDATES = 1000;  // 1000 帧 = 10 秒 @ 0.01 fixedDeltaTime
    private const float COM_STD_THRESHOLD = 1e-4f;  // 标准差阈值
    private const float COM_DISTANCE_THRESHOLD = 0.01f;  // 平均距离阈值 1cm

    private RobotRLAgent agent;
    private ArticulationBody[] arts;

    /// <summary>
    /// 加载真实场景并获取真实的 RobotRLAgent
    /// </summary>
    [UnitySetUp]
    public IEnumerator SetUp()
    {
        // 加载真实场景
#if UNITY_EDITOR
        string scenePath = "Assets/Playground/ShipDeck.unity";
        var loadParams = new LoadSceneParameters(LoadSceneMode.Single);
        yield return EditorSceneManager.LoadSceneAsyncInPlayMode(scenePath, loadParams);
#else
        yield return SceneManager.LoadSceneAsync("ShipDeck", LoadSceneMode.Single);
#endif

        // 等待场景加载完成
        yield return null;
        yield return null;

        // 查找 RobotRLAgent（使用真实场景中的机器人）
        agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent！请确认 ShipDeck.unity 场景包含机器人");

        // 获取 ArticulationBody 数组（通过反射或 public 方法）
        // 假设 RobotRLAgent 有 GetArticulationBodies() 方法
        // 如果没有，需要在 R3 中添加
        arts = agent.GetComponentsInChildren<ArticulationBody>();
        Assert.Greater(arts.Length, 0, "机器人应包含 ArticulationBody 组件");

        Debug.Log($"[CoMVerificationTest] 场景加载完成");
        Debug.Log($"[CoMVerificationTest] 找到机器人: {agent.name}");
        Debug.Log($"[CoMVerificationTest] ArticulationBody 数量: {arts.Length}");

        // 等待物理初始化
        yield return new WaitForFixedUpdate();
        yield return new WaitForFixedUpdate();
    }

    /// <summary>
    /// Test (a): 整机质量验证
    /// </summary>
    [UnityTest]
    public IEnumerator TestWholeBodyMassVerification()
    {
        yield return null;  // 确保 SetUp 已完成

        // 计算总质量
        float totalMass = 0f;
        List<string> massReport = new List<string>();

        massReport.Add("=== 整机质量验证 ===");
        for (int i = 0; i < arts.Length; i++)
        {
            if (arts[i] != null)
            {
                totalMass += arts[i].mass;
                massReport.Add($"  Link {i+1:D2}: {arts[i].name,-35} mass={arts[i].mass:F3} kg");
            }
        }

        massReport.Add("---");
        massReport.Add($"总质量: {totalMass:F3} kg");
        massReport.Add($"标称质量 (Unitree G1): {UNITREE_G1_NOMINAL_MASS:F3} kg");
        massReport.Add($"ArticulationBody 数量: {arts.Length}");

        float deviation = Mathf.Abs(totalMass - UNITREE_G1_NOMINAL_MASS);
        float deviationPercent = (deviation / UNITREE_G1_NOMINAL_MASS) * 100f;
        massReport.Add($"偏差: {deviation:F3} kg ({deviationPercent:F1}%)");

        // 输出完整报告
        foreach (var line in massReport)
        {
            Debug.Log($"[CoMVerificationTest] {line}");
        }

        // 断言
        Assert.LessOrEqual(deviationPercent, MASS_TOLERANCE_PERCENT,
            $"质量偏差 ({deviationPercent:F1}%) 超过容忍度 ({MASS_TOLERANCE_PERCENT}%)。" +
            $"实际质量: {totalMass:F3} kg, 标称质量: {UNITREE_G1_NOMINAL_MASS:F3} kg");

        Debug.Log($"[CoMVerificationTest] ✓ 质量验证通过");
    }

    /// <summary>
    /// Test (b): 质心 x/z 随时间变化验证
    /// </summary>
    [UnityTest]
    public IEnumerator TestCoMChangesOverTime()
    {
        yield return null;

        // 获取根节点
        ArticulationBody rootBody = arts[0];
        Transform bodyTransform = rootBody.transform;

        // 采样质心位置
        List<Vector2> comSamples = new List<Vector2>();  // (x, z) in local coordinates

        for (int i = 0; i < TEST_FIXEDUPDATES; i++)
        {
            yield return new WaitForFixedUpdate();

            // 计算全身质心（调用真实实现）
            Vector3 wholeBodyCoM = agent.GetWholeBodyCoM();
            Vector3 wholeBodyCoMLocal = bodyTransform.InverseTransformPoint(wholeBodyCoM);

            comSamples.Add(new Vector2(wholeBodyCoMLocal.x, wholeBodyCoMLocal.z));
        }

        // 计算标准差
        float meanX = comSamples.Average(v => v.x);
        float meanZ = comSamples.Average(v => v.y);

        float varianceX = comSamples.Average(v => Mathf.Pow(v.x - meanX, 2));
        float varianceZ = comSamples.Average(v => Mathf.Pow(v.y - meanZ, 2));

        float stdX = Mathf.Sqrt(varianceX);
        float stdZ = Mathf.Sqrt(varianceZ);

        Debug.Log($"[CoMVerificationTest] 质心变化统计（{TEST_FIXEDUPDATES} 帧）:");
        Debug.Log($"[CoMVerificationTest]   CoM X: mean={meanX:F5}, std={stdX:F5}");
        Debug.Log($"[CoMVerificationTest]   CoM Z: mean={meanZ:F5}, std={stdZ:F5}");

        // 断言：至少一个方向的标准差 > 阈值
        float maxStd = Mathf.Max(stdX, stdZ);
        Assert.Greater(maxStd, COM_STD_THRESHOLD,
            $"质心应随时间变化！X 标准差={stdX:E}, Z 标准差={stdZ:E}, 阈值={COM_STD_THRESHOLD:E}。" +
            $"这表明质心计算可能是恒定的（P0-1 bug）");

        Debug.Log($"[CoMVerificationTest] ✓ 质心变化验证通过（最大标准差={maxStd:E}）");
    }

    /// <summary>
    /// Test (c): 单 link 质心 vs 全身质心距离验证
    /// </summary>
    [UnityTest]
    public IEnumerator TestRootCoMVsWholeBodyCoM()
    {
        yield return null;

        ArticulationBody rootBody = arts[0];

        // 采样多帧求平均距离
        List<float> distances = new List<float>();

        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();

            Vector3 rootCoM = rootBody.worldCenterOfMass;
            Vector3 wholeBodyCoM = agent.GetWholeBodyCoM();
            float distance = Vector3.Distance(rootCoM, wholeBodyCoM);

            distances.Add(distance);
        }

        float avgDistance = distances.Average();
        float maxDistance = distances.Max();
        float minDistance = distances.Min();

        Debug.Log($"[CoMVerificationTest] 根节点质心 vs 全身质心距离:");
        Debug.Log($"[CoMVerificationTest]   平均距离: {avgDistance:F5} m");
        Debug.Log($"[CoMVerificationTest]   最大距离: {maxDistance:F5} m");
        Debug.Log($"[CoMVerificationTest]   最小距离: {minDistance:F5} m");

        // 断言：多 link 机器人的根节点质心应与全身质心有差异
        if (arts.Length > 1)
        {
            Assert.Greater(avgDistance, COM_DISTANCE_THRESHOLD,
                $"多 link 机器人的根节点质心应与全身质心有差异！" +
                $"平均距离={avgDistance:F5} m, 阈值={COM_DISTANCE_THRESHOLD} m。" +
                $"这表明代码未正确计算全身质心（P0-1 bug）");
        }

        Debug.Log($"[CoMVerificationTest] ✓ 质心差异验证通过");
    }

    // T3: 移除本地实现，改用 RobotRLAgent.GetWholeBodyCoM() 公共接口
}
