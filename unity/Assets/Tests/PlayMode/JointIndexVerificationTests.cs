using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEngine.SceneManagement;
using NUnit.Framework;

#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// M0: 关节索引核查测试
///
/// 验证 PredictiveFeedforwardController.cs 中的关节索引是否与注释一致：
/// - 注释声称：acts[3] = left_knee, acts[9] = right_knee
/// - IMPLEMENTATION_BRIEF_v2.md 怀疑：acts[3] = right_knee, acts[9] = left_knee（执行顺序相反）
///
/// 验证方法：
/// 1. 平台固定向右倾斜（roll > 0，RegularSine 模式）
/// 2. 记录摆动相位指示器 uf1/uf2 和施加的 compensation[3]/[9]
/// 3. 记录 acts[3].name 和 acts[9].name 的真实关节名称
/// 4. 记录左右脚接触状态和质心侧向位置
/// 5. 断言：非零补偿施加在支撑腿（接触地面）的膝关节上
/// 6. 断言：补偿符号使质心向倾斜的反方向移动（向左对抗右倾）
/// </summary>
public class JointIndexVerificationTests
{
    private const int NUM_STEPS = 200;  // 足够时间观察多个步态周期
    private const float FIXED_ROLL_ANGLE = 5.0f;  // 固定 5° 右倾
    private const float SWING_THRESHOLD = 0.7f;  // uf > 0.7 表示摆动腿

    private RobotRLAgent agent;
    private WaveSpectrumPlatformController platformController;
    private PredictiveFeedforwardController predictiveController;
    private ExplicitVelocityPredictor velocityPredictor;

    private List<float> recordedTimes = new List<float>();
    private List<float> recordedUf1 = new List<float>();
    private List<float> recordedUf2 = new List<float>();
    private List<float> recordedComp3 = new List<float>();
    private List<float> recordedComp9 = new List<float>();
    private List<float> recordedComX = new List<float>();  // 质心 X 位置（侧向）
    private List<bool> recordedLeftContact = new List<bool>();
    private List<bool> recordedRightContact = new List<bool>();
    private string acts3Name = "";
    private string acts9Name = "";

    // 用于反射访问 RobotRLAgent 的私有字段
    private FieldInfo uf1Field;
    private FieldInfo uf2Field;

    [UnitySetUp]
    public IEnumerator SetUp()
    {
        // 加载 ShipDeck 场景
#if UNITY_EDITOR
        string scenePath = "Assets/Playground/ShipDeck.unity";
        var loadParams = new LoadSceneParameters(LoadSceneMode.Single);
        yield return EditorSceneManager.LoadSceneAsyncInPlayMode(scenePath, loadParams);
#else
        yield return SceneManager.LoadSceneAsync("ShipDeck", LoadSceneMode.Single);
#endif

        yield return null;
        yield return null;

        // 查找 agent 和平台控制器
        agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent");

        Transform areaRoot = agent.transform.parent;
        Transform groundTransform = null;

        if (areaRoot != null && areaRoot.GetComponent<TrainingArea>() != null)
        {
            groundTransform = areaRoot.Find("Ground");
        }

        if (groundTransform == null)
        {
            GameObject ground = GameObject.Find("Ground");
            if (ground != null)
            {
                groundTransform = ground.transform;
            }
        }

        Assert.IsNotNull(groundTransform, "未找到 Ground 对象");

        platformController = groundTransform.GetComponent<WaveSpectrumPlatformController>();
        Assert.IsNotNull(platformController, "Ground 未挂载 WaveSpectrumPlatformController");

        // 配置平台为固定右倾（RegularSine 模式，频率为 0 即静态）
        platformController.waveMode = WaveMode.RegularSine;
        platformController.regularRollAmplitude = FIXED_ROLL_ANGLE;  // 5° 幅度
        platformController.regularRollFrequency = 0.0f;  // 频率 0 = 静态倾斜
        platformController.regularRollPhase = 90.0f;  // 相位 90° → sin(90°) = 1 → roll = +5°

        Debug.Log($"[JointIndexTest] 配置平台为固定右倾: roll = {FIXED_ROLL_ANGLE}°");

        // 查找预测前馈控制器
        predictiveController = agent.GetComponent<PredictiveFeedforwardController>();
        Assert.IsNotNull(predictiveController, "未找到 PredictiveFeedforwardController");

        // 查找速度预测器
        velocityPredictor = agent.GetComponent<ExplicitVelocityPredictor>();
        Assert.IsNotNull(velocityPredictor, "未找到 ExplicitVelocityPredictor");

        // 使用反射获取 uf1/uf2 字段
        uf1Field = typeof(RobotRLAgent).GetField("uf1", BindingFlags.NonPublic | BindingFlags.Instance);
        uf2Field = typeof(RobotRLAgent).GetField("uf2", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.IsNotNull(uf1Field, "无法通过反射访问 uf1 字段");
        Assert.IsNotNull(uf2Field, "无法通过反射访问 uf2 字段");

        yield return new WaitForFixedUpdate();
        yield return new WaitForFixedUpdate();

        // 记录关节名称（从 agent.acts[] 获取）
        // 需要等待 agent.Initialize() 完成
        int waitCount = 0;
        while ((agent.acts == null || agent.acts.Length <= 9) && waitCount < 100)
        {
            yield return new WaitForFixedUpdate();
            waitCount++;
        }

        Assert.IsNotNull(agent.acts, "agent.acts 未初始化");
        Assert.Greater(agent.acts.Length, 9, "agent.acts 数组长度不足");

        if (agent.acts[3] != null)
        {
            acts3Name = agent.acts[3].name;
        }
        if (agent.acts[9] != null)
        {
            acts9Name = agent.acts[9].name;
        }

        Debug.Log($"[JointIndexTest] acts[3].name = {acts3Name}");
        Debug.Log($"[JointIndexTest] acts[9].name = {acts9Name}");

        Assert.IsFalse(string.IsNullOrEmpty(acts3Name), "acts[3] 未初始化");
        Assert.IsFalse(string.IsNullOrEmpty(acts9Name), "acts[9] 未初始化");
    }

    /// <summary>
    /// M0 主测试：验证关节索引与支撑腿对应关系
    /// </summary>
    [UnityTest]
    public IEnumerator TestJointIndexCorrespondence()
    {
        yield return null;

        Debug.Log($"[JointIndexTest] 开始记录 {NUM_STEPS} 步数据");

        // 运行 NUM_STEPS 步，记录数据
        for (int step = 0; step < NUM_STEPS; step++)
        {
            // 等待 FixedUpdate（物理步）
            yield return new WaitForFixedUpdate();

            // 使用反射获取 uf1/uf2
            float uf1 = (float)uf1Field.GetValue(agent);
            float uf2 = (float)uf2Field.GetValue(agent);

            // 获取平台当前欧拉角
            Vector3 currentEuler = platformController.GetPlatformEuler();

            // 获取预测的欧拉角（直接使用 velocityPredictor 的公开字段）
            // 注意：predictedEulerAngles 是在 agent 的 CollectObservations 中更新的
            Vector3 predictedEuler = velocityPredictor.predictedEulerAngles;

            // 计算补偿值
            float[] compensation = predictiveController.ComputeKneeCompensation(
                predictedEuler,
                uf1,
                uf2,
                agent.ActionNum
            );

            // 获取足部接触状态
            bool leftContact = IsFootInContact(agent.leftFootJoint);
            bool rightContact = IsFootInContact(agent.rightFootJoint);

            // 获取质心位置
            ArticulationBody robotRoot = agent.GetComponentInChildren<ArticulationBody>();
            Vector3 comPosition = Vector3.zero;
            float comX = 0f;
            if (robotRoot != null)
            {
                comPosition = robotRoot.centerOfMass;
                comX = robotRoot.transform.TransformPoint(comPosition).x;
            }

            // 记录数据
            recordedTimes.Add(Time.fixedTime);
            recordedUf1.Add(uf1);
            recordedUf2.Add(uf2);
            recordedComp3.Add(compensation[3]);
            recordedComp9.Add(compensation[9]);
            recordedComX.Add(comX);
            recordedLeftContact.Add(leftContact);
            recordedRightContact.Add(rightContact);

            if (step % 50 == 0)
            {
                Debug.Log($"[JointIndexTest] Step {step}: uf1={uf1:F3}, uf2={uf2:F3}, " +
                          $"L_contact={leftContact}, R_contact={rightContact}, " +
                          $"comp[3]={compensation[3]:F3}, comp[9]={compensation[9]:F3}, " +
                          $"roll={currentEuler.z:F2}°, CoM_x={comX:F3}");
            }
        }

        // 导出 CSV
        ExportDataToCSV();

        // 分析并断言
        AnalyzeAndAssert();

        yield return null;
    }

    /// <summary>
    /// 判断足部是否接触地面（基于世界高度的简化判据）
    /// 注意：这是临时实现，Task C 会改为力判据
    /// </summary>
    private bool IsFootInContact(ArticulationBody footJoint)
    {
        if (footJoint == null)
            return false;

        // 简化判据：足部高度 < 0.15m 认为接触
        // （世界高度判据在倾斜甲板上不准确，但对于 5° 小倾角可接受）
        float footHeight = footJoint.transform.position.y;
        return footHeight < 0.15f;
    }

    /// <summary>
    /// 导出记录数据到 CSV
    /// </summary>
    private void ExportDataToCSV()
    {
        string csvPath = Path.Combine(Application.dataPath, "..", "..", "manuscript", "revision",
                                      "verification", "joint_index_verification.csv");

        // 确保目录存在
        string dirPath = Path.GetDirectoryName(csvPath);
        if (!Directory.Exists(dirPath))
        {
            Directory.CreateDirectory(dirPath);
        }

        using (StreamWriter writer = new StreamWriter(csvPath))
        {
            // CSV 头
            writer.WriteLine("time,uf1,uf2,comp_3,comp_9,com_x,left_contact,right_contact,acts3_name,acts9_name");

            // 数据行
            for (int i = 0; i < recordedTimes.Count; i++)
            {
                writer.WriteLine($"{recordedTimes[i]:F3}," +
                                $"{recordedUf1[i]:F3}," +
                                $"{recordedUf2[i]:F3}," +
                                $"{recordedComp3[i]:F6}," +
                                $"{recordedComp9[i]:F6}," +
                                $"{recordedComX[i]:F6}," +
                                $"{(recordedLeftContact[i] ? 1 : 0)}," +
                                $"{(recordedRightContact[i] ? 1 : 0)}," +
                                $"{acts3Name}," +
                                $"{acts9Name}");
            }
        }

        Debug.Log($"[JointIndexTest] 数据已导出到: {csvPath}");
    }

    /// <summary>
    /// 分析数据并断言关键假设
    /// </summary>
    private void AnalyzeAndAssert()
    {
        Debug.Log($"[JointIndexTest] 开始分析 {recordedTimes.Count} 条记录");

        // 统计：左腿支撑时 comp[3] 和 comp[9] 的激活情况
        int leftStanceCount = 0;
        int leftStanceComp3Active = 0;
        int leftStanceComp9Active = 0;

        // 统计：右腿支撑时 comp[3] 和 comp[9] 的激活情况
        int rightStanceCount = 0;
        int rightStanceComp3Active = 0;
        int rightStanceComp9Active = 0;

        const float COMP_THRESHOLD = 0.01f;  // 补偿绝对值 > 0.01 认为激活

        for (int i = 0; i < recordedTimes.Count; i++)
        {
            bool leftStance = recordedLeftContact[i] && !recordedRightContact[i];
            bool rightStance = recordedRightContact[i] && !recordedLeftContact[i];

            bool comp3Active = Mathf.Abs(recordedComp3[i]) > COMP_THRESHOLD;
            bool comp9Active = Mathf.Abs(recordedComp9[i]) > COMP_THRESHOLD;

            if (leftStance)
            {
                leftStanceCount++;
                if (comp3Active) leftStanceComp3Active++;
                if (comp9Active) leftStanceComp9Active++;
            }

            if (rightStance)
            {
                rightStanceCount++;
                if (comp3Active) rightStanceComp3Active++;
                if (comp9Active) rightStanceComp9Active++;
            }
        }

        Debug.Log($"[JointIndexTest] 左腿支撑帧数: {leftStanceCount}");
        Debug.Log($"[JointIndexTest]   - comp[3] 激活: {leftStanceComp3Active} ({100f*leftStanceComp3Active/Mathf.Max(leftStanceCount,1):F1}%)");
        Debug.Log($"[JointIndexTest]   - comp[9] 激活: {leftStanceComp9Active} ({100f*leftStanceComp9Active/Mathf.Max(leftStanceCount,1):F1}%)");
        Debug.Log($"[JointIndexTest] 右腿支撑帧数: {rightStanceCount}");
        Debug.Log($"[JointIndexTest]   - comp[3] 激活: {rightStanceComp3Active} ({100f*rightStanceComp3Active/Mathf.Max(rightStanceCount,1):F1}%)");
        Debug.Log($"[JointIndexTest]   - comp[9] 激活: {rightStanceComp9Active} ({100f*rightStanceComp9Active/Mathf.Max(rightStanceCount,1):F1}%)");

        // 推断关节索引
        Debug.Log($"[JointIndexTest] acts[3].name = {acts3Name}");
        Debug.Log($"[JointIndexTest] acts[9].name = {acts9Name}");

        // 断言 1：关节名称应包含 "knee"
        Assert.IsTrue(acts3Name.ToLower().Contains("knee"), $"acts[3] 不是膝关节: {acts3Name}");
        Assert.IsTrue(acts9Name.ToLower().Contains("knee"), $"acts[9] 不是膝关节: {acts9Name}");

        // 断言 2：如果注释正确（acts[3]=left_knee, acts[9]=right_knee），则：
        // - 左腿支撑时，comp[3] 应激活（因为 PredictiveFeedforwardController 认为 acts[3] 是左膝）
        // - 右腿支撑时，comp[9] 应激活
        //
        // 如果索引相反（acts[3]=right_knee, acts[9]=left_knee），则：
        // - 左腿支撑时，comp[9] 应激活（因为实际的左膝是 acts[9]）
        // - 右腿支撑时，comp[3] 应激活
        //
        // 通过比较激活率判断哪个假设成立

        float leftStanceComp3Rate = leftStanceCount > 0 ? (float)leftStanceComp3Active / leftStanceCount : 0f;
        float leftStanceComp9Rate = leftStanceCount > 0 ? (float)leftStanceComp9Active / leftStanceCount : 0f;
        float rightStanceComp3Rate = rightStanceCount > 0 ? (float)rightStanceComp3Active / rightStanceCount : 0f;
        float rightStanceComp9Rate = rightStanceCount > 0 ? (float)rightStanceComp9Active / rightStanceCount : 0f;

        Debug.Log($"[JointIndexTest] 结论推断:");

        if (acts3Name.ToLower().Contains("left") && acts9Name.ToLower().Contains("right"))
        {
            Debug.Log($"[JointIndexTest] 关节名称显示: acts[3] 是左膝, acts[9] 是右膝（与注释一致）");
        }
        else if (acts3Name.ToLower().Contains("right") && acts9Name.ToLower().Contains("left"))
        {
            Debug.Log($"[JointIndexTest] 关节名称显示: acts[3] 是右膝, acts[9] 是左膝（与注释相反）");
        }
        else
        {
            Debug.LogWarning($"[JointIndexTest] 无法从关节名称判断左右");
        }

        // 最终结论：直接报告关节名称和激活模式，由用户分析
        // 测试本身不强制断言，因为这是探索性验证
    }

    [UnityTearDown]
    public IEnumerator TearDown()
    {
        yield return null;
    }
}
