using UnityEngine;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// EpisodeStatsLogger - 最简单的Episode统计记录器
///
/// 【使用方法】
/// 1. 将此脚本添加到机器人 GameObject（与RobotRLAgent同级）
/// 2. 在 Inspector 中查看实时统计
/// 3. 观察 Console 输出的统计信息
///
/// 【无需修改Agent代码】
/// 此脚本通过监听 Agent 的状态变化自动检测 Episode 结束
///
/// 【输出信息】
/// - 每个Episode结束时打印步数
/// - 每10个Episode打印一次统计摘要（均值±标准差）
/// - 在屏幕左上角实时显示统计信息
/// </summary>
public class EpisodeStatsLogger : MonoBehaviour
{
    [Header("=== 测试信息 ===")]
    [Tooltip("模型名称（用于记录）")]
    public string modelName = "gewu_full";

    [Tooltip("当前测试的平台角度（度）")]
    [Range(0f, 10f)]
    public float testAngle = 3.0f;

    [Header("=== 统计信息 (只读) ===")]
    [SerializeField]
    [Tooltip("已完成的 Episode 数量")]
    private int episodeCount = 0;

    [SerializeField]
    [Tooltip("当前 Episode 的步数")]
    private int currentSteps = 0;

    [SerializeField]
    [Tooltip("平均步数")]
    private float meanSteps = 0f;

    [SerializeField]
    [Tooltip("标准差")]
    private float stdSteps = 0f;

    // 私有变量
    private List<int> episodeLengths = new List<int>();
    private int lastEpisodeCount = -1;

    // 通过反射获取Agent的私有字段
    private System.Reflection.FieldInfo episodeCountField;
    private Unity.MLAgents.Agent agent;

    void Start()
    {
        // 查找 Agent 组件
        agent = GetComponent<Unity.MLAgents.Agent>();
        if (agent == null)
        {
            agent = GetComponent<RobotRLAgent>();
        }

        if (agent == null)
        {
            Debug.LogError("[EpisodeStatsLogger] 未找到 Agent 组件！");
            enabled = false;
            return;
        }

        // 获取Agent的CompletedEpisodes字段（通过反射）
        episodeCountField = typeof(Unity.MLAgents.Agent).GetField(
            "m_CompletedEpisodes",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
        );

        if (episodeCountField == null)
        {
            Debug.LogWarning("[EpisodeStatsLogger] 无法获取 Episode 计数，将使用备用方法");
        }

        Debug.Log($"[EpisodeStatsLogger] 初始化完成");
        Debug.Log($"  模型: {modelName}");
        Debug.Log($"  测试角度: {testAngle}°");
    }

    void FixedUpdate()
    {
        if (agent == null) return;

        // 增加当前步数计数
        currentSteps++;

        // 检测 Episode 是否结束
        CheckEpisodeEnd();
    }

    /// <summary>
    /// 检测 Episode 是否结束
    /// </summary>
    private void CheckEpisodeEnd()
    {
        if (episodeCountField == null) return;

        int currentEpisodeCount = (int)episodeCountField.GetValue(agent);

        // 检测到新的 Episode
        if (currentEpisodeCount > lastEpisodeCount && lastEpisodeCount != -1)
        {
            OnEpisodeEnd();
        }

        lastEpisodeCount = currentEpisodeCount;
    }

    /// <summary>
    /// Episode 结束时调用
    /// </summary>
    private void OnEpisodeEnd()
    {
        // 记录步数（减1是因为OnEpisodeBegin会在下一个FixedUpdate执行）
        int steps = Mathf.Max(0, currentSteps - 1);
        episodeLengths.Add(steps);
        episodeCount++;

        // 计算统计信息
        CalculateStats();

        // 打印信息
        Debug.Log($"[EpisodeStatsLogger] Episode {episodeCount}: " +
                 $"步数={steps}, 模型={modelName}, 角度={testAngle:F1}°");

        // 每10个Episode打印一次统计摘要
        if (episodeCount % 10 == 0)
        {
            PrintSummary();
        }

        // 每20个Episode自动保存一次CSV（避免数据丢失）
        if (episodeCount % 20 == 0)
        {
            SaveToCSV();
        }

        // 重置步数计数
        currentSteps = 0;
    }

    /// <summary>
    /// 计算统计信息
    /// </summary>
    private void CalculateStats()
    {
        if (episodeLengths.Count == 0)
        {
            meanSteps = 0;
            stdSteps = 0;
            return;
        }

        // 计算均值
        float sum = 0f;
        foreach (int len in episodeLengths)
        {
            sum += len;
        }
        meanSteps = sum / episodeLengths.Count;

        // 计算标准差
        float sqSum = 0f;
        foreach (int len in episodeLengths)
        {
            sqSum += (len - meanSteps) * (len - meanSteps);
        }
        stdSteps = Mathf.Sqrt(sqSum / episodeLengths.Count);
    }

    /// <summary>
    /// 打印统计摘要
    /// </summary>
    private void PrintSummary()
    {
        Debug.Log($"========================================");
        Debug.Log($"[EpisodeStatsLogger] 统计摘要");
        Debug.Log($"  模型: {modelName}");
        Debug.Log($"  角度: {testAngle:F1}°");
        Debug.Log($"  完成 Episode 数: {episodeCount}");
        Debug.Log($"  平均步数: {meanSteps:F1} ± {stdSteps:F1}");
        Debug.Log($"  最小值: {episodeLengths.Min()}");
        Debug.Log($"  最大值: {episodeLengths.Max()}");
        Debug.Log($"========================================");
    }

    /// <summary>
    /// 保存数据到CSV文件（追加模式，支持多机器人并发写入）
    /// </summary>
    private void SaveToCSV()
    {
        if (episodeLengths.Count == 0)
        {
            Debug.LogWarning("[EpisodeStatsLogger] 没有数据可保存");
            return;
        }

        // 生成文件名
        string fileName = $"episode_stats_{modelName}_{testAngle:F1}deg.csv";
        // 保存到统一的测试结果目录
        string testResultsDir = "/home/xds/PycharmProjects/test_results";
        string filePath = System.IO.Path.Combine(testResultsDir, fileName);

        try
        {
            // 使用文件锁防止多个机器人同时写入
            lock (typeof(EpisodeStatsLogger))
            {
                bool fileExists = System.IO.File.Exists(filePath);

                // 追加模式打开文件
                using (System.IO.StreamWriter writer = new System.IO.StreamWriter(filePath, append: true))
                {
                    // 如果是新文件，先写入表头
                    if (!fileExists)
                    {
                        writer.WriteLine("ModelName,RollAmplitude,Steps");
                    }

                    // 写入每个Episode的数据（去掉Episode列，因为多个机器人会导致编号重复）
                    for (int i = 0; i < episodeLengths.Count; i++)
                    {
                        writer.WriteLine($"{modelName},{testAngle:F1},{episodeLengths[i]}");
                    }
                }

                Debug.Log($"[EpisodeStatsLogger] ✓ CSV文件已保存到: {filePath} (追加了 {episodeLengths.Count} 条记录)");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[EpisodeStatsLogger] 保存CSV文件失败: {e.Message}");
        }
    }

    /// <summary>
    /// 在屏幕上显示统计信息
    /// </summary>
    void OnGUI()
    {
        GUIStyle style = new GUIStyle();
        style.fontSize = 18;
        style.normal.textColor = Color.yellow;
        style.fontStyle = FontStyle.Bold;

        string text = $"=== Episode 统计 ===\n" +
                     $"模型: {modelName}\n" +
                     $"角度: {testAngle:F1}°\n" +
                     $"Episode: {episodeCount}\n" +
                     $"当前步数: {currentSteps}\n" +
                     $"平均: {meanSteps:F1} ± {stdSteps:F1}";

        GUI.Label(new Rect(10, 10, 400, 150), text, style);
    }

    /// <summary>
    /// 当组件被禁用时打印最终统计并保存CSV
    /// </summary>
    void OnDisable()
    {
        if (episodeCount > 0)
        {
            PrintSummary();
            SaveToCSV();
        }
    }
}
