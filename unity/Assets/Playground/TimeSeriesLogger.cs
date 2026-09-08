using UnityEngine;
using System.IO;
using System.Text;

/// <summary>
/// 时间序列数据记录器
/// 用于记录机器人运行过程中的关键数据，供论文实验分析使用
///
/// 使用方法：
/// 1. 将此脚本挂载到Robot对象上（与RobotRLAgent同级）
/// 2. 在Inspector中勾选 enableLogging 开启记录
/// 3. 设置 logDuration（建议10-20秒，即1000-2000步）
/// 4. 运行推理模式，数据会自动保存到 results/time_series/ 目录
/// </summary>
public class TimeSeriesLogger : MonoBehaviour
{
    [Header("=== 数据记录配置 ===")]
    [Tooltip("是否启用数据记录（建议仅在推理模式下启用）")]
    public bool enableLogging = false;

    [Tooltip("记录持续时间（秒），0=记录整个Episode")]
    public float logDuration = 10f;

    [Tooltip("记录起始延迟（秒），用于跳过初始不稳定阶段")]
    public float logStartDelay = 2f;

    [Tooltip("记录频率（每N个FixedUpdate记录一次，1=每帧记录）")]
    [Range(1, 10)]
    public int recordEveryNFrames = 1;

    [Header("=== 模型标识 ===")]
    [Tooltip("当前实验的模型名称（用于文件命名）")]
    public string modelName = "ours_full";

    [Tooltip("平台摆幅标识（用于文件命名）")]
    public string platformConfig = "roll5deg";

    // 组件引用
    private RobotRLAgent agent;
    private ExplicitVelocityPredictor predictor;
    private PredictiveFeedforwardController feedforwardController;
    private PlatformMotionController platformController;
    private ArticulationBody robotBody;
    private ArticulationBody leftKneeJoint;
    private ArticulationBody rightKneeJoint;

    // 记录状态
    private bool isRecording = false;
    private float recordingStartTime = 0f;
    private int frameCounter = 0;
    private StringBuilder csvData;
    private string outputPath;

    void Start()
    {
        if (!enableLogging)
        {
            Debug.Log("[TimeSeriesLogger] 数据记录已禁用");
            enabled = false;
            return;
        }

        // 获取组件引用
        agent = GetComponent<RobotRLAgent>();
        if (agent == null)
        {
            Debug.LogError("[TimeSeriesLogger] 未找到RobotRLAgent组件！");
            enabled = false;
            return;
        }

        // 通过反射获取私有字段（或者修改RobotRLAgent添加公共访问器）
        predictor = GetComponent<ExplicitVelocityPredictor>();
        feedforwardController = GetComponent<PredictiveFeedforwardController>();

        // 查找平台控制器
        GameObject ground = GameObject.Find("Ground");
        if (ground != null)
        {
            platformController = ground.GetComponent<PlatformMotionController>();
        }

        // 查找机器人关节（支持嵌套层级）
        // P0-5 修复：移除硬编码路径，使用动态查找根 ArticulationBody
        ArticulationBody[] allBodies = GetComponentsInChildren<ArticulationBody>();
        if (allBodies.Length > 0)
        {
            robotBody = allBodies[0];  // 第一个通常是根节点（pelvis）
            Debug.Log($"[TimeSeriesLogger] 找到根节点: {robotBody.name}");
        }
        else
        {
            Debug.LogError("[TimeSeriesLogger] 未找到任何 ArticulationBody！");
        }

        leftKneeJoint = FindJoint("left_knee");
        rightKneeJoint = FindJoint("right_knee");

        // 初始化CSV数据
        csvData = new StringBuilder();
        csvData.AppendLine(GetCSVHeader());

        // 创建输出目录
        string resultsDir = Path.Combine(Application.dataPath, "Playground", "results", "time_series");
        Directory.CreateDirectory(resultsDir);

        // 生成文件名（带时间戳）
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string fileName = $"{modelName}_{platformConfig}_{timestamp}.csv";
        outputPath = Path.Combine(resultsDir, fileName);

        Debug.Log($"[TimeSeriesLogger] 已初始化，输出文件: {outputPath}");
        Debug.Log($"[TimeSeriesLogger] 将在 {logStartDelay}秒后开始记录，持续 {logDuration}秒");
    }

    void FixedUpdate()
    {
        if (!enableLogging || !isRecording)
        {
            // 检查是否到达开始记录的时间
            if (!isRecording && Time.time >= logStartDelay)
            {
                isRecording = true;
                recordingStartTime = Time.time;
                Debug.Log($"[TimeSeriesLogger] 开始记录数据... (t={Time.time:F2}s)");
            }
            return;
        }

        // 检查是否超过记录时长
        float elapsedTime = Time.time - recordingStartTime;
        if (logDuration > 0 && elapsedTime > logDuration)
        {
            StopRecording();
            return;
        }

        // 按频率记录
        frameCounter++;
        if (frameCounter % recordEveryNFrames != 0)
        {
            return;
        }

        // 收集数据并写入CSV
        RecordCurrentFrame(elapsedTime);
    }

    void RecordCurrentFrame(float elapsedTime)
    {
        // 基础时间信息
        float timestamp = Time.time;
        int stepCount = (int)(elapsedTime / Time.fixedDeltaTime);

        // 1. 机器人姿态和速度
        Vector3 robotEuler = robotBody != null ? robotBody.transform.eulerAngles : Vector3.zero;
        Vector3 robotVel = robotBody != null ? robotBody.velocity : Vector3.zero;
        Vector3 robotAngVel = robotBody != null ? robotBody.angularVelocity : Vector3.zero;

        // 将欧拉角转换到[-180, 180]范围
        float robotRoll = NormalizeAngle(robotEuler.z);
        float robotPitch = NormalizeAngle(robotEuler.x);
        float robotYaw = NormalizeAngle(robotEuler.y);

        // 2. 预测值（如果存在）
        Vector3 predVel = Vector3.zero;
        Vector3 predAngVel = Vector3.zero;
        Vector3 predEuler = Vector3.zero;
        float predRoll = 0f;

        if (predictor != null)
        {
            predictor.GetSimplifiedPredictionObservations(out predVel, out predAngVel, out predEuler);
            predRoll = NormalizeAngle(predEuler.z);
        }

        // 3. 前馈补偿量（如果存在）
        // P0-2 修复：记录实际施加的补偿值，而非理论值
        float leftKneeComp = 0f;
        float rightKneeComp = 0f;
        float filteredRollError = 0f;

        if (feedforwardController != null)
        {
            filteredRollError = feedforwardController.GetFilteredRollError();
        }

        if (agent != null)
        {
            // 获取实际施加的补偿值（包含相位门控和饱和限制）
            float[] actualComp = agent.GetLastPredictiveCompensation();
            if (actualComp != null && actualComp.Length > 9)
            {
                leftKneeComp = actualComp[3];   // left_knee index
                rightKneeComp = actualComp[9];  // right_knee index
            }
        }

        // 4. 平台状态（如果存在）
        float platformRoll = 0f;
        float platformPitch = 0f;
        float platformHeave = 0f;

        if (platformController != null)
        {
            platformRoll = NormalizeAngle(platformController.currentEulerAngles.z);
            platformPitch = NormalizeAngle(platformController.currentEulerAngles.x);
            platformHeave = platformController.currentPositionOffset.y;
        }

        // 5. 关节实际角度
        float leftKneeAngle = 0f;
        float rightKneeAngle = 0f;

        if (leftKneeJoint != null)
        {
            leftKneeAngle = leftKneeJoint.jointPosition[0] * Mathf.Rad2Deg;
        }
        if (rightKneeJoint != null)
        {
            rightKneeAngle = rightKneeJoint.jointPosition[0] * Mathf.Rad2Deg;
        }

        // 6. CoM位置（如果RobotRLAgent启用了CoM观测）
        // P0-1 修复：使用全身合质心而非单个 link 的质心
        Vector3 comPosition = Vector3.zero;
        if (agent != null)
        {
            Vector3 worldCoM = ComputeWholeBodyCoM();
            comPosition = robotBody.transform.InverseTransformPoint(worldCoM);
        }

        // 构建CSV行
        csvData.AppendLine(
            $"{timestamp:F3}," +
            $"{stepCount}," +
            $"{robotRoll:F3},{robotPitch:F3},{robotYaw:F3}," +
            $"{robotVel.x:F3},{robotVel.y:F3},{robotVel.z:F3}," +
            $"{robotAngVel.x:F3},{robotAngVel.y:F3},{robotAngVel.z:F3}," +
            $"{predRoll:F3}," +
            $"{leftKneeComp:F3},{rightKneeComp:F3}," +
            $"{filteredRollError:F3}," +
            $"{platformRoll:F3},{platformPitch:F3},{platformHeave:F3}," +
            $"{leftKneeAngle:F3},{rightKneeAngle:F3}," +
            $"{comPosition.x:F3},{comPosition.y:F3},{comPosition.z:F3}"
        );
    }

    void StopRecording()
    {
        if (!isRecording) return;

        isRecording = false;

        // 保存CSV文件
        try
        {
            File.WriteAllText(outputPath, csvData.ToString());
            Debug.Log($"[TimeSeriesLogger] 数据已保存: {outputPath}");
            Debug.Log($"[TimeSeriesLogger] 共记录 {csvData.ToString().Split('\n').Length - 2} 个数据点");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"[TimeSeriesLogger] 保存文件失败: {e.Message}");
        }

        // 禁用此组件（避免重复记录）
        enabled = false;
    }

    void OnApplicationQuit()
    {
        // 确保退出时保存数据
        if (isRecording)
        {
            StopRecording();
        }
    }

    string GetCSVHeader()
    {
        return "timestamp,step," +
               "robot_roll,robot_pitch,robot_yaw," +
               "robot_vel_x,robot_vel_y,robot_vel_z," +
               "robot_angvel_x,robot_angvel_y,robot_angvel_z," +
               "predicted_roll," +
               "left_knee_compensation,right_knee_compensation," +
               "filtered_roll_error," +
               "platform_roll,platform_pitch,platform_heave," +
               "left_knee_angle,right_knee_angle," +
               "com_x,com_y,com_z";
    }

    /// <summary>
    /// P0-5 修复：使用 Contains 模糊匹配关节名称，提高兼容性
    /// </summary>
    ArticulationBody FindJoint(string jointName)
    {
        ArticulationBody[] allJoints = GetComponentsInChildren<ArticulationBody>();
        foreach (var joint in allJoints)
        {
            // 改用 Contains 匹配，例如 "left_knee" 可以匹配 "left_knee_link" 或 "G1_left_knee"
            if (joint.name.Contains(jointName))
            {
                Debug.Log($"[TimeSeriesLogger] 找到关节: '{jointName}' -> '{joint.name}'");
                return joint;
            }
        }
        // P0-5：找不到时明确报错，而非静默返回 null
        Debug.LogError($"[TimeSeriesLogger] ⚠️ 未找到包含 '{jointName}' 的关节！可用关节列表：");
        foreach (var joint in allJoints)
        {
            Debug.LogError($"  - {joint.name}");
        }
        return null;
    }

    float NormalizeAngle(float angle)
    {
        // 将角度从[0, 360]转换到[-180, 180]
        if (angle > 180f)
            return angle - 360f;
        return angle;
    }

    /// <summary>
    /// 计算全身合质心（正确的方法）
    /// P0-1 修复：替代原来只用单个 link 的 worldCenterOfMass 的错误方法
    /// </summary>
    Vector3 ComputeWholeBodyCoM()
    {
        ArticulationBody[] allBodies = GetComponentsInChildren<ArticulationBody>();
        float totalMass = 0f;
        Vector3 weightedSum = Vector3.zero;

        foreach (var ab in allBodies)
        {
            if (ab == null) continue;
            weightedSum += ab.worldCenterOfMass * ab.mass;
            totalMass += ab.mass;
        }

        if (totalMass <= 0f)
        {
            Debug.LogError("[TimeSeriesLogger] 计算质心时总质量为0！");
            return Vector3.zero;
        }

        return weightedSum / totalMass;
    }

    // 在Inspector中显示记录状态
    void OnGUI()
    {
        if (!enableLogging) return;

        GUIStyle style = new GUIStyle();
        style.fontSize = 20;
        style.normal.textColor = isRecording ? Color.red : Color.yellow;

        if (isRecording)
        {
            float elapsed = Time.time - recordingStartTime;
            float remaining = logDuration - elapsed;
            GUI.Label(new Rect(10, 10, 400, 30),
                     $"● 正在记录数据... {elapsed:F1}s / {logDuration:F1}s",
                     style);
        }
        else if (Time.time < logStartDelay)
        {
            GUI.Label(new Rect(10, 10, 400, 30),
                     $"⏳ 等待开始记录... {logStartDelay - Time.time:F1}s",
                     style);
        }
    }
}
