using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// V1: Feedforward Sign A/B Test Logger
///
/// 自动运行 10 个 episode（seed 1001-1010），记录每个 episode 的:
/// - 存活步数
/// - Posture RMSE
/// - Wave seed
///
/// 用法:
/// 1. 挂载到 G1 机器人上（与 RobotRLAgent 同级）
/// 2. 设置 testGroup = A 或 B
/// 3. Play 运行，自动完成 10 个 episode
/// 4. 结果保存到 Assets/Playground/results/feedforward_ab_test_{group}.csv
/// </summary>
public class FeedforwardABTestLogger : MonoBehaviour
{
    [Header("=== V1 A/B Test Configuration ===")]
    [Tooltip("测试组: A (positive sign) 或 B (negative sign)")]
    public string testGroup = "A";

    [Tooltip("每组测试的 episode 数量")]
    public int episodesPerGroup = 10;

    [Tooltip("起始 seed")]
    public int startSeed = 1001;

    [Header("=== Runtime Stats (Read Only) ===")]
    [SerializeField] private int currentEpisode = 0;
    [SerializeField] private int currentSeed = 0;
    [SerializeField] private int currentSteps = 0;
    [SerializeField] private float meanSteps = 0f;

    // Data collection
    private List<EpisodeData> episodeDataList = new List<EpisodeData>();
    private int lastEpisodeStepCount = -1;
    private RobotRLAgent agent;
    private WaveSpectrumPlatformController waveController;

    // Posture tracking
    private float cumulativePostureError = 0f;
    private int postureErrorSamples = 0;

    private class EpisodeData
    {
        public int Seed;
        public int Steps;
        public float PostureRMSE;
        public float CumulativeReward;
    }

    void Start()
    {
        agent = GetComponent<RobotRLAgent>();
        if (agent == null)
        {
            Debug.LogError("[FeedforwardABTestLogger] RobotRLAgent not found!");
            enabled = false;
            return;
        }

        // Find WaveSpectrumPlatformController
        waveController = FindObjectOfType<WaveSpectrumPlatformController>();
        if (waveController == null)
        {
            Debug.LogError("[FeedforwardABTestLogger] WaveSpectrumPlatformController not found!");
            enabled = false;
            return;
        }

        currentEpisode = 0;
        currentSeed = startSeed;

        Debug.Log($"[V1 A/B Test] Starting Group {testGroup} test with {episodesPerGroup} episodes");
        Debug.Log($"[V1 A/B Test] Seed range: {startSeed} to {startSeed + episodesPerGroup - 1}");

        // Set first seed
        SetWaveSeed(currentSeed);
    }

    void FixedUpdate()
    {
        if (agent == null) return;

        // Track posture error
        Vector3 torsoEuler = agent.transform.rotation.eulerAngles;
        float roll = NormalizeAngle(torsoEuler.z);
        float pitch = NormalizeAngle(torsoEuler.x);
        float postureError = Mathf.Sqrt(roll * roll + pitch * pitch);

        cumulativePostureError += postureError * postureError;
        postureErrorSamples++;

        // Detect episode end
        int currentSteps = agent.StepCount;
        if (lastEpisodeStepCount > 0 && currentSteps < lastEpisodeStepCount)
        {
            // Episode ended
            OnEpisodeEnd(lastEpisodeStepCount);
        }
        lastEpisodeStepCount = currentSteps;
        this.currentSteps = currentSteps;
    }

    void OnEpisodeEnd(int steps)
    {
        // Calculate posture RMSE for this episode
        float postureRMSE = 0f;
        if (postureErrorSamples > 0)
        {
            postureRMSE = Mathf.Sqrt(cumulativePostureError / postureErrorSamples);
        }

        // Get cumulative reward
        float cumulativeReward = agent.GetCumulativeReward();

        // Record data
        episodeDataList.Add(new EpisodeData
        {
            Seed = currentSeed,
            Steps = steps,
            PostureRMSE = postureRMSE,
            CumulativeReward = cumulativeReward
        });

        Debug.Log($"[V1 A/B Test] Group {testGroup} | Seed {currentSeed} | Episode {currentEpisode + 1}/{episodesPerGroup} " +
                  $"| Steps: {steps} | Posture RMSE: {postureRMSE:F3} | Reward: {cumulativeReward:F2}");

        // Reset posture tracking
        cumulativePostureError = 0f;
        postureErrorSamples = 0;

        currentEpisode++;

        if (currentEpisode >= episodesPerGroup)
        {
            // Test complete
            OnTestComplete();
        }
        else
        {
            // Move to next seed
            currentSeed++;
            SetWaveSeed(currentSeed);
        }
    }

    void OnTestComplete()
    {
        Debug.Log($"[V1 A/B Test] Group {testGroup} test complete!");

        // Calculate statistics
        float meanSteps = (float)episodeDataList.Average(e => e.Steps);
        float stdSteps = Mathf.Sqrt((float)episodeDataList.Average(e => Mathf.Pow(e.Steps - meanSteps, 2)));
        float meanRMSE = (float)episodeDataList.Average(e => e.PostureRMSE);

        Debug.Log($"[V1 A/B Test] Group {testGroup} Results:");
        Debug.Log($"  Mean Steps: {meanSteps:F1} ± {stdSteps:F1}");
        Debug.Log($"  Mean Posture RMSE: {meanRMSE:F3}");

        this.meanSteps = meanSteps;

        // Save to CSV
        SaveResultsToCSV();

        Debug.Log($"[V1 A/B Test] Results saved to Assets/Playground/results/feedforward_ab_test_{testGroup}.csv");
        Debug.Log("[V1 A/B Test] ===== TEST COMPLETE =====");

        // Stop playmode
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
        #endif
    }

    void SaveResultsToCSV()
    {
        string resultsDir = Path.Combine(Application.dataPath, "Playground", "results");
        if (!Directory.Exists(resultsDir))
        {
            Directory.CreateDirectory(resultsDir);
        }

        string csvPath = Path.Combine(resultsDir, $"feedforward_ab_test_{testGroup}.csv");

        using (StreamWriter writer = new StreamWriter(csvPath))
        {
            // Header
            writer.WriteLine("Group,Seed,Episode,Steps,PostureRMSE,CumulativeReward");

            // Data
            for (int i = 0; i < episodeDataList.Count; i++)
            {
                var data = episodeDataList[i];
                writer.WriteLine($"{testGroup},{data.Seed},{i + 1},{data.Steps},{data.PostureRMSE:F4},{data.CumulativeReward:F2}");
            }
        }
    }

    void SetWaveSeed(int seed)
    {
        if (waveController != null)
        {
            waveController.waveSeed = seed;
            Debug.Log($"[V1 A/B Test] Set wave seed to {seed}");
        }
    }

    float NormalizeAngle(float angle)
    {
        if (angle > 180f) angle -= 360f;
        return angle;
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle();
        style.fontSize = 16;
        style.normal.textColor = Color.white;

        string status = currentEpisode < episodesPerGroup ? "RUNNING" : "COMPLETE";
        Color statusColor = currentEpisode < episodesPerGroup ? Color.green : Color.cyan;

        GUI.color = statusColor;
        GUI.Label(new Rect(10, 10, 500, 30), $"V1 A/B Test - Group {testGroup} - {status}", style);
        GUI.color = Color.white;
        GUI.Label(new Rect(10, 35, 500, 25), $"Episode: {currentEpisode + 1}/{episodesPerGroup} | Seed: {currentSeed} | Steps: {currentSteps}", style);

        if (currentEpisode > 0)
        {
            GUI.Label(new Rect(10, 60, 500, 25), $"Mean Steps: {meanSteps:F1}", style);
        }
    }
}
