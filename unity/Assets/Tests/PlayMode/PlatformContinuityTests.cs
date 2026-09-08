using System.Collections;
using System.Collections.Generic;
using System.Linq;  // H3: For Distinct()
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;
using NUnit.Framework;

#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// G3: 平台连续性测试（Training Area 架构）
///
/// 验证 Training Area 架构下的平台行为：
/// - 同一 area 内：episode 边界处 ResampleWave + ResetMotion 后仍无尖峰（连续性）
/// - 多 area 间：不同 area 的海况参数独立（验证随机化）
/// </summary>
public class PlatformContinuityTests
{
    private const float ANGULAR_VELOCITY_SPIKE_THRESHOLD = 1.5f;  // 1.5x 解析角速度
    private const int NUM_EPISODES = 3;  // 测试 3 个 episode 边界
    private const int STEPS_PER_EPISODE = 100;  // 每个 episode 100 步

    private RobotRLAgent agent;
    private WaveSpectrumPlatformController platformController;

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

        yield return null;
        yield return null;

        // G3: 查找 agent 和对应的 platform
        agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent");

        // I3: 使用 TrainingArea 标记组件识别 area（替代字符串启发式）
        Transform areaRoot = agent.transform.parent;
        Transform groundTransform = null;
        string areaName = "Root (single agent)";

        if (areaRoot != null && areaRoot.GetComponent<TrainingArea>() != null)
        {
            // Training Area 架构：在 area 中查找
            groundTransform = areaRoot.Find("Ground");
            areaName = areaRoot.name;
        }

        if (groundTransform == null)
        {
            // Fallback: 全局查找（单 agent 场景或测试场景）
            GameObject ground = GameObject.Find("Ground");
            if (ground != null)
            {
                groundTransform = ground.transform;
            }
        }

        Assert.IsNotNull(groundTransform, "未找到 Ground 对象");

        platformController = groundTransform.GetComponent<WaveSpectrumPlatformController>();
        Assert.IsNotNull(platformController, $"Ground 未挂载 WaveSpectrumPlatformController");

        // H3/I3: 遍历所有 Training Area 统一配置（支持多 area 测试）
        RobotRLAgent[] allAgents = Object.FindObjectsByType<RobotRLAgent>(FindObjectsSortMode.None);
        int configuredAreas = 0;

        foreach (var a in allAgents)
        {
            Transform root = a.transform.parent;
            // I3: 使用 TrainingArea 标记组件识别 area
            if (root != null && root.GetComponent<TrainingArea>() != null)
            {
                Transform groundTrans = root.Find("Ground");
                if (groundTrans != null)
                {
                    var ctrl = groundTrans.GetComponent<WaveSpectrumPlatformController>();
                    if (ctrl != null)
                    {
                        // H3: 使用真实范围（不是单点值）
                        ctrl.waveMode = WaveMode.JONSWAP;
                        ctrl.significantRollAmplitudeRange = new Vector2(2f, 5f);  // 真实范围
                        ctrl.peakPeriodRange = new Vector2(6f, 10f);               // 真实范围
                        ctrl.ResampleWave(999 + configuredAreas);  // 不同 area 不同种子
                        configuredAreas++;
                    }
                }
            }
        }

        // 如果只有一个 area（单 agent 场景），配置主平台
        if (configuredAreas == 0)
        {
            platformController.waveMode = WaveMode.JONSWAP;
            platformController.significantRollAmplitudeRange = new Vector2(2f, 5f);
            platformController.peakPeriodRange = new Vector2(6f, 10f);
            platformController.ResampleWave(999);
            configuredAreas = 1;
        }

        Debug.Log($"[PlatformContinuityTest] 场景加载完成，配置了 {configuredAreas} 个 area，测试主 area: {areaName}");

        yield return new WaitForFixedUpdate();
        yield return new WaitForFixedUpdate();
    }

    /// <summary>
    /// G3/H2/J2: 验证同一 area 内 episode 边界处平台运动连续（无尖峰）
    ///
    /// J2 修正：修复 ill-posed 判据
    /// 1. 分母改为 episode 内 |ω_analytic| 的最大值（固定尺度）
    /// 2. 添加机器人基座速度变化判据（更直接）
    /// </summary>
    [UnityTest]
    public IEnumerator TestNoAngularVelocitySpikesAtEpisodeBoundaries()
    {
        yield return null;

        // J2: 获取机器人基座（root ArticulationBody）
        ArticulationBody robotRoot = agent.GetComponent<ArticulationBody>();
        if (robotRoot == null)
        {
            robotRoot = agent.GetComponentInChildren<ArticulationBody>();
        }
        Assert.IsNotNull(robotRoot, "未找到机器人基座 ArticulationBody");

        for (int episode = 0; episode < NUM_EPISODES; episode++)
        {
            Debug.Log($"[PlatformContinuityTest] Episode {episode + 1}/{NUM_EPISODES}");

            // J2: 记录整个 episode 的解析角速度最大值
            float maxAnalyticalVel = 0f;
            List<float> numericalVels = new List<float>();
            List<Vector3> robotVelocities = new List<Vector3>();
            List<bool> isBoundary = new List<bool>();

            Vector3 lastEuler = platformController.GetPlatformEuler();
            float lastTime = Time.fixedTime;

            // 运行 episode
            for (int step = 0; step < STEPS_PER_EPISODE; step++)
            {
                yield return new WaitForFixedUpdate();

                // 采样平台状态
                Vector3 currentEuler = platformController.GetPlatformEuler();
                float currentTime = Time.fixedTime;
                Vector3 analyticalVelocity = platformController.GetPlatformAngularVelocity();

                // J2: 更新解析角速度最大值
                maxAnalyticalVel = Mathf.Max(maxAnalyticalVel, Mathf.Abs(analyticalVelocity.z));

                // 计算数值角速度（roll 轴）
                float deltaTime = currentTime - lastTime;
                if (deltaTime > 1e-6f)
                {
                    float deltaRoll = currentEuler.z - lastEuler.z;
                    if (deltaRoll > 180f) deltaRoll -= 360f;
                    if (deltaRoll < -180f) deltaRoll += 360f;

                    float numericalVelocity = deltaRoll / deltaTime;
                    numericalVels.Add(numericalVelocity);
                    robotVelocities.Add(robotRoot.velocity);
                    isBoundary.Add(step >= STEPS_PER_EPISODE - 3 || step < 3);  // J2: 边界前后 3 步
                }

                lastEuler = currentEuler;
                lastTime = currentTime;
            }

            // J2 判据 1: 数值角速度 vs 解析角速度最大值
            float maxNumericalVelAtBoundary = 0f;
            for (int i = 0; i < numericalVels.Count; i++)
            {
                if (isBoundary[i])
                {
                    maxNumericalVelAtBoundary = Mathf.Max(maxNumericalVelAtBoundary, Mathf.Abs(numericalVels[i]));
                }
            }

            float spike = maxNumericalVelAtBoundary / Mathf.Max(maxAnalyticalVel, 1e-3f);

            Debug.Log($"[PlatformContinuityTest] Episode {episode}:");
            Debug.Log($"  解析角速度最大值: {maxAnalyticalVel:F2} deg/s");
            Debug.Log($"  边界数值角速度峰值: {maxNumericalVelAtBoundary:F2} deg/s");
            Debug.Log($"  尖峰比例: {spike:F2}x");

            Assert.LessOrEqual(spike, ANGULAR_VELOCITY_SPIKE_THRESHOLD,
                $"Episode {episode} 边界角速度尖峰 ({spike:F2}x) 超过阈值 ({ANGULAR_VELOCITY_SPIKE_THRESHOLD:F2}x)");

            // J2 判据 2: 机器人基座速度变化
            List<float> velocityChanges = new List<float>();
            for (int i = 1; i < robotVelocities.Count; i++)
            {
                float velChange = (robotVelocities[i] - robotVelocities[i - 1]).magnitude;
                velocityChanges.Add(velChange);
            }

            float maxVelChangeAtBoundary = 0f;
            float maxVelChangeInside = 0f;

            for (int i = 0; i < velocityChanges.Count; i++)
            {
                if (isBoundary[i])
                {
                    maxVelChangeAtBoundary = Mathf.Max(maxVelChangeAtBoundary, velocityChanges[i]);
                }
                else
                {
                    maxVelChangeInside = Mathf.Max(maxVelChangeInside, velocityChanges[i]);
                }
            }

            float velChangeRatio = maxVelChangeAtBoundary / Mathf.Max(maxVelChangeInside, 1e-6f);

            Debug.Log($"  机器人基座速度变化（边界）: {maxVelChangeAtBoundary:F4} m/s");
            Debug.Log($"  机器人基座速度变化（内部）: {maxVelChangeInside:F4} m/s");
            Debug.Log($"  速度变化比例: {velChangeRatio:F2}x");

            Assert.LessOrEqual(velChangeRatio, ANGULAR_VELOCITY_SPIKE_THRESHOLD,
                $"Episode {episode} 边界速度变化 ({velChangeRatio:F2}x) 超过阈值 ({ANGULAR_VELOCITY_SPIKE_THRESHOLD:F2}x)");

            // G3: Episode 结束，触发 OnEpisodeBegin（调用 ResampleWave + ResetMotion）
            agent.EndEpisode();
            yield return new WaitForFixedUpdate();
        }

        Debug.Log($"[PlatformContinuityTest] ✓ 平台连续性验证通过");
    }

    /// <summary>
    /// G3: 验证多个 Training Area 的海况参数独立
    ///
    /// 如果场景中有多个 area（14个 clone），验证它们在同一时刻的海况参数不同。
    /// 这确保每个 agent 的训练数据分布独立。
    /// </summary>
    [UnityTest]
    [Ignore("J5: 需要训练时的多 area 环境（14个 clone）。当前 ShipDeck.unity 只有单 area。")]
    public IEnumerator TestMultiAreaSeaStateIndependence()
    {
        yield return null;

        // 查找所有 RobotRLAgent
        RobotRLAgent[] allAgents = Object.FindObjectsByType<RobotRLAgent>(FindObjectsSortMode.None);

        if (allAgents.Length < 2)
        {
            Debug.LogWarning("[PlatformContinuityTest] 场景中只有 1 个 agent，跳过多 area 独立性测试");
            yield break;  // 测试通过（不适用）
        }

        Debug.Log($"[PlatformContinuityTest] 发现 {allAgents.Length} 个 Training Area");

        // 收集每个 area 的平台控制器
        List<WaveSpectrumPlatformController> platforms = new List<WaveSpectrumPlatformController>();

        foreach (var agent in allAgents)
        {
            Transform areaRoot = agent.transform.parent;
            if (areaRoot != null)
            {
                Transform groundTransform = areaRoot.Find("Ground");
                if (groundTransform != null)
                {
                    var platform = groundTransform.GetComponent<WaveSpectrumPlatformController>();
                    if (platform != null)
                    {
                        platforms.Add(platform);
                    }
                }
            }
        }

        Assert.GreaterOrEqual(platforms.Count, 2, "未找到至少 2 个平台控制器");

        // 触发所有 agent 的 episode 结束（会调用 ResampleWave）
        foreach (var agent in allAgents)
        {
            agent.EndEpisode();
        }

        yield return new WaitForFixedUpdate();
        yield return new WaitForFixedUpdate();

        // H3: 检查海况参数（Hs, Tp, seed）是否互不相同（强断言）
        List<float> significantAmplitudes = new List<float>();
        List<float> peakPeriods = new List<float>();
        List<int> seeds = new List<int>();

        for (int i = 0; i < platforms.Count; i++)
        {
            significantAmplitudes.Add(platforms[i].CurrentSignificantRollAmplitude);
            peakPeriods.Add(platforms[i].CurrentPeakPeriod);
            seeds.Add(platforms[i].CurrentSeed);

            Debug.Log($"[PlatformContinuityTest] Area {i}: " +
                     $"Hs={platforms[i].CurrentSignificantRollAmplitude:F3}°, " +
                     $"Tp={platforms[i].CurrentPeakPeriod:F3}s, " +
                     $"seed={platforms[i].CurrentSeed}");
        }

        // H3: 断言各 area 的 Hs、Tp、seed 三者互不相同
        // 由于 ResampleWave 时使用范围随机，不同 area 的参数应该不同
        int uniqueHs = significantAmplitudes.Distinct().Count();
        int uniqueTp = peakPeriods.Distinct().Count();
        int uniqueSeeds = seeds.Distinct().Count();

        Assert.AreEqual(platforms.Count, uniqueSeeds,
            $"所有 area 的 seed 应该互不相同，但只有 {uniqueSeeds} 个唯一值（共 {platforms.Count} 个 area）");

        // Hs 和 Tp 可能偶然相同（连续随机值），但概率极低
        // 如果所有 area 的 Hs 或 Tp 完全相同，说明随机化失败
        Assert.Greater(uniqueHs, 1,
            $"所有 area 的显著横摇幅度完全相同 ({significantAmplitudes[0]:F3}°)，说明随机化失败");

        Assert.Greater(uniqueTp, 1,
            $"所有 area 的谱峰周期完全相同 ({peakPeriods[0]:F3}s)，说明随机化失败");

        Debug.Log($"[PlatformContinuityTest] ✓ 多 area 海况独立性验证通过: " +
                 $"{uniqueSeeds} 个唯一 seed, {uniqueHs} 个唯一 Hs, {uniqueTp} 个唯一 Tp");
    }
}
