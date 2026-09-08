using UnityEngine;
using Unity.MLAgents;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// RobotPerformanceLogger - 模型测试性能记录器
///
/// 【功能说明】
/// 用于测试训练好的模型在不同平台倾角下的表现
/// 统计每个Episode的存活步数、累积奖励等指标
///
/// 【使用方法】
/// 1. 将此脚本挂载到机器人 GameObject 上（与 RobotRLAgent 同级）
/// 2. 在 Inspector 中设置参数：
///    - Test Mode: 选择测试模式
///      * Fixed Angle: 固定角度测试
///      * Multiple Angles: 多角度自动测试
///    - Fixed Roll Amplitude: 固定角度模式下的测试角度
///    - Test Angles: 多角度模式下的测试角度列表
///    - Episodes Per Test: 每个角度测试多少个 Episode
/// 3. 在 Unity 中加载训练好的模型（在 Agent 组件中设置 .onnx 文件）
/// 4. 点击 Play 按钮开始测试
/// 5. 测试完成后，结果会保存到 Assets/Playground/test_results.csv
///
/// 【输出格式】
/// CSV 文件包含以下列：
/// - ModelName: 模型名称
/// - RollAmplitude: 平台横摇幅度（度）
/// - Episode: Episode 编号
/// - Steps: 存活步数
/// - CumulativeReward: 累积奖励
/// - FallReason: 失败原因（Pitch/Roll超限或超时）
///
/// 【注意事项】
/// - 必须配合 PlatformMotionController 使用
/// - 建议在 Inspector 中关闭课程学习（useCurriculumLearning = false）
/// - 测试时建议设置 Time.timeScale = 10 加速测试
/// </summary>
public class RobotPerformanceLogger : MonoBehaviour
{
    [Header("=== 测试模式配置 ===")]
    [Tooltip("测试模式：固定角度 或 多角度自动测试")]
    public TestMode testMode = TestMode.FixedAngle;

    public enum TestMode
    {
        FixedAngle,      // 固定角度：手动设置 PlatformMotionController 的角度
        MultipleAngles   // 多角度：自动遍历测试多个角度
    }

    [Header("=== 固定角度模式 ===")]
    [Tooltip("固定角度模式下的测试角度（度）")]
    [Range(0f, 10f)]
    public float fixedRollAmplitude = 3.0f;

    [Header("=== 多角度模式 ===")]
    [Tooltip("多角度模式下的测试角度列表（度）\n建议：0, 1, 2, 3, 4, 5")]
    public float[] testAngles = new float[] { 0f, 1f, 2f, 3f, 4f, 5f };

    [Tooltip("每个角度测试多少个 Episode\n建议：10-50")]
    [Range(5, 100)]
    public int episodesPerTest = 20;

    [Header("=== 模型信息 ===")]
    [Tooltip("模型名称（用于记录）\n例如：gewu_full, gewu_nocom, gewu_baseline")]
    public string modelName = "gewu_full";

    [Header("=== 输出设置 ===")]
    [Tooltip("结果保存路径（相对于 Assets 文件夹）")]
    public string outputPath = "Playground/test_results.csv";

    [Tooltip("是否在每次测试开始时清空之前的结果")]
    public bool clearPreviousResults = false;

    [Header("=== 运行时信息 (只读) ===")]
    [SerializeField]
    [Tooltip("当前测试的角度")]
    private float currentTestAngle = 0f;

    [SerializeField]
    [Tooltip("当前角度已完成的 Episode 数")]
    private int currentAngleEpisodeCount = 0;

    [SerializeField]
    [Tooltip("总共完成的 Episode 数")]
    private int totalEpisodeCount = 0;

    [SerializeField]
    [Tooltip("测试是否已完成")]
    private bool testCompleted = false;

    // 私有变量
    private Agent agent;
    private PlatformMotionController platformController;
    private int currentStepCount = 0;
    private float currentCumulativeReward = 0f;
    private List<TestResult> results = new List<TestResult>();
    private int currentAngleIndex = 0;
    private bool isTestRunning = false;
    private string fullOutputPath;

    // 数据结构
    private struct TestResult
    {
        public string modelName;
        public float rollAmplitude;
        public int episode;
        public int steps;
        public float cumulativeReward;
        public string fallReason;
    }

    void Start()
    {
        // 查找 Agent 组件
        agent = GetComponent<RobotRLAgent>();

        if (agent == null)
        {
            Debug.LogError("[RobotPerformanceLogger] 未找到 RobotRLAgent 组件！");
            enabled = false;
            return;
        }

        // 查找平台控制器
        GameObject ground = GameObject.Find("Ground");
        if (ground != null)
        {
            platformController = ground.GetComponent<PlatformMotionController>();
        }

        if (platformController == null)
        {
            Debug.LogError("[RobotPerformanceLogger] 未找到 PlatformMotionController！");
            enabled = false;
            return;
        }

        // 设置输出路径
        fullOutputPath = Path.Combine(Application.dataPath, outputPath);

        // 清空之前的结果（可选）
        if (clearPreviousResults && File.Exists(fullOutputPath))
        {
            File.Delete(fullOutputPath);
            Debug.Log($"[RobotPerformanceLogger] 已清空之前的测试结果");
        }

        // 初始化测试
        if (testMode == TestMode.MultipleAngles)
        {
            StartMultiAngleTest();
        }
        else
        {
            StartFixedAngleTest();
        }
    }

    /// <summary>
    /// 固定角度测试模式
    /// </summary>
    private void StartFixedAngleTest()
    {
        currentTestAngle = fixedRollAmplitude;
        platformController.rollAmplitude = fixedRollAmplitude;
        platformController.useCurriculumLearning = false;
        isTestRunning = true;

        Debug.Log($"[RobotPerformanceLogger] 开始固定角度测试");
        Debug.Log($"  模型: {modelName}");
        Debug.Log($"  角度: {fixedRollAmplitude}°");
        Debug.Log($"  测试次数: {episodesPerTest}");
    }

    /// <summary>
    /// 多角度测试模式
    /// </summary>
    private void StartMultiAngleTest()
    {
        if (testAngles == null || testAngles.Length == 0)
        {
            Debug.LogError("[RobotPerformanceLogger] 测试角度列表为空！");
            enabled = false;
            return;
        }

        currentAngleIndex = 0;
        currentTestAngle = testAngles[0];
        platformController.rollAmplitude = currentTestAngle;
        platformController.useCurriculumLearning = false;
        isTestRunning = true;

        Debug.Log($"[RobotPerformanceLogger] 开始多角度测试");
        Debug.Log($"  模型: {modelName}");
        Debug.Log($"  角度列表: {string.Join(", ", testAngles)}°");
        Debug.Log($"  每个角度测试次数: {episodesPerTest}");
        Debug.Log($"  总 Episode 数: {testAngles.Length * episodesPerTest}");
    }

    void FixedUpdate()
    {
        if (!isTestRunning) return;

        currentStepCount++;
    }

    /// <summary>
    /// 监听 Agent 的奖励变化
    /// </summary>
    void LateUpdate()
    {
        if (!isTestRunning) return;

        // 获取累积奖励（通过反射访问私有字段）
        var rewardField = typeof(Agent).GetField("m_CumulativeReward",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (rewardField != null)
        {
            currentCumulativeReward = (float)rewardField.GetValue(agent);
        }
    }

    /// <summary>
    /// 在 Episode 结束时记录结果
    /// 需要在 RobotRLAgent 的 OnEpisodeBegin() 中调用 LogEpisodeEnd()
    /// </summary>
    public void LogEpisodeEnd(string fallReason)
    {
        if (!isTestRunning) return;

        // 记录结果
        TestResult result = new TestResult
        {
            modelName = modelName,
            rollAmplitude = currentTestAngle,
            episode = currentAngleEpisodeCount + 1,
            steps = currentStepCount,
            cumulativeReward = currentCumulativeReward,
            fallReason = fallReason
        };
        results.Add(result);

        Debug.Log($"[RobotPerformanceLogger] Episode {totalEpisodeCount + 1} 完成: " +
                  $"角度={currentTestAngle:F1}°, 步数={currentStepCount}, 奖励={currentCumulativeReward:F2}, 原因={fallReason}");

        // 重置计数器
        currentStepCount = 0;
        currentCumulativeReward = 0f;
        currentAngleEpisodeCount++;
        totalEpisodeCount++;

        // 检查是否完成当前角度的测试
        if (currentAngleEpisodeCount >= episodesPerTest)
        {
            OnAngleTestCompleted();
        }
    }

    /// <summary>
    /// 当前角度测试完成
    /// </summary>
    private void OnAngleTestCompleted()
    {
        Debug.Log($"[RobotPerformanceLogger] 角度 {currentTestAngle:F1}° 测试完成 ({episodesPerTest} Episodes)");

        // 保存当前结果
        SaveResults();

        // 检查是否还有更多角度需要测试
        if (testMode == TestMode.MultipleAngles)
        {
            currentAngleIndex++;
            if (currentAngleIndex < testAngles.Length)
            {
                // 切换到下一个角度
                currentTestAngle = testAngles[currentAngleIndex];
                platformController.rollAmplitude = currentTestAngle;
                currentAngleEpisodeCount = 0;
                Debug.Log($"[RobotPerformanceLogger] 切换到角度 {currentTestAngle:F1}°");
            }
            else
            {
                // 所有测试完成
                OnAllTestsCompleted();
            }
        }
        else
        {
            // 固定角度模式：测试完成
            OnAllTestsCompleted();
        }
    }

    /// <summary>
    /// 所有测试完成
    /// </summary>
    private void OnAllTestsCompleted()
    {
        isTestRunning = false;
        testCompleted = true;

        Debug.Log($"[RobotPerformanceLogger] ========================================");
        Debug.Log($"[RobotPerformanceLogger] 所有测试已完成！");
        Debug.Log($"[RobotPerformanceLogger] 总 Episode 数: {totalEpisodeCount}");
        Debug.Log($"[RobotPerformanceLogger] 结果已保存到: {fullOutputPath}");
        Debug.Log($"[RobotPerformanceLogger] ========================================");

        // 打印统计摘要
        PrintSummary();
    }

    /// <summary>
    /// 保存结果到 CSV 文件
    /// </summary>
    private void SaveResults()
    {
        bool fileExists = File.Exists(fullOutputPath);

        using (StreamWriter writer = new StreamWriter(fullOutputPath, append: true))
        {
            // 写入 CSV 表头（仅在文件不存在时）
            if (!fileExists)
            {
                writer.WriteLine("ModelName,RollAmplitude,Episode,Steps,CumulativeReward,FallReason");
            }

            // 写入所有结果
            foreach (var result in results)
            {
                writer.WriteLine($"{result.modelName},{result.rollAmplitude:F1},{result.episode}," +
                               $"{result.steps},{result.cumulativeReward:F4},{result.fallReason}");
            }
        }

        Debug.Log($"[RobotPerformanceLogger] 已保存 {results.Count} 条结果到 {fullOutputPath}");
        results.Clear(); // 清空已保存的结果
    }

    /// <summary>
    /// 打印测试摘要
    /// </summary>
    private void PrintSummary()
    {
        if (results.Count == 0 && File.Exists(fullOutputPath))
        {
            // 从文件读取结果进行统计
            Debug.Log($"[RobotPerformanceLogger] 统计摘要（基于已保存的结果）：");
            Debug.Log($"  详细数据请查看: {fullOutputPath}");
        }
    }

    /// <summary>
    /// 在 Inspector 中显示进度
    /// </summary>
    void OnGUI()
    {
        if (!isTestRunning && !testCompleted) return;

        GUIStyle style = new GUIStyle();
        style.fontSize = 20;
        style.normal.textColor = Color.white;

        if (isTestRunning)
        {
            string progressText = $"测试进行中...\n" +
                                 $"模型: {modelName}\n" +
                                 $"当前角度: {currentTestAngle:F1}°\n" +
                                 $"完成: {currentAngleEpisodeCount}/{episodesPerTest} Episodes\n" +
                                 $"总进度: {totalEpisodeCount}/{GetTotalEpisodes()} Episodes";
            GUI.Label(new Rect(10, 10, 400, 150), progressText, style);
        }
        else if (testCompleted)
        {
            style.normal.textColor = Color.green;
            string completedText = $"测试完成！\n" +
                                  $"总 Episode 数: {totalEpisodeCount}\n" +
                                  $"结果已保存到: {outputPath}";
            GUI.Label(new Rect(10, 10, 600, 100), completedText, style);
        }
    }

    /// <summary>
    /// 获取总 Episode 数
    /// </summary>
    private int GetTotalEpisodes()
    {
        if (testMode == TestMode.MultipleAngles)
        {
            return testAngles.Length * episodesPerTest;
        }
        else
        {
            return episodesPerTest;
        }
    }
}
