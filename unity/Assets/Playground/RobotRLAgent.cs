using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using Unity.Sentis;


public class RobotRLAgent : Agent
{
    int tp = 0;
    int tq = 0;
    int tt = 0;
    public bool fixbody = false;
    public bool train;

    [Header("=== X1(b): Runtime Cloning Control ===")]
    [Tooltip("X1(b): 是否在运行时克隆 TrainingArea（默认 false，使用 --num-envs 多进程）")]
    public bool spawnClonesAtRuntime = false;

    [Header("=== Ablation Study Configuration ===")]
    [Tooltip("启用质心和足部接触观测（+5维）\nBaseline=false, +CoM=true")]
    public bool enableCoMObservation = true;

    [Tooltip("启用显式预测观测（+24维）\nBaseline=false, +Pred=true")]
    public bool enablePredictionObservation = true;

    [Tooltip("启用预测前馈控制（不改变观测维度）\nOurs=true, 其他=false")]
    public bool enableFeedforwardControl = true;

    // F1 修复：使用新的波浪谱平台控制器
    private WaveSpectrumPlatformController platformController;

    // 足部关节引用（用于接触检测）
    public ArticulationBody leftFootJoint;  // M0: 临时改为 public 供测试访问
    public ArticulationBody rightFootJoint;  // M0: 临时改为 public 供测试访问

    // C: 接触力检测（主判据）
    [Header("接触检测参数")]
    [Tooltip("接触力阈值（N），体重的5%约16N")]
    public float contactForceThreshold = 16f;

    [Tooltip("相对平台高度阈值（m），备用判据")]
    public float contactHeightThreshold = 0.05f;

    // 接触力缓存（由 OnCollisionStay 更新）
    private float leftFootContactForce = 0f;
    private float rightFootContactForce = 0f;

    // C: Posture RMSE 累积（episode 内）
    private float rollErrorSumSquared = 0f;
    private float pitchErrorSumSquared = 0f;
    private int postureStepCount = 0;

    // 显式速度预测器
    private ExplicitVelocityPredictor velocityPredictor;

    // M1: 甲板 IMU (真前馈传感器)
    private DeckIMU deckIMU;

    // 预测前馈控制器
    private PredictiveFeedforwardController predictiveController;

    // P0-2 修复：缓存实际施加的前馈补偿值（用于日志记录）
    private float[] lastPredictiveComp = new float[16];

    // T2: 观测收集错误计数（不使用early-return，避免0维观测污染训练）
    private int observationErrorCount = 0;
    private bool isDestroying = false;

    float uff = 0;
    float uf1 = 0;
    float uf2 = 0;
    float[] u = new float[16] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] ut = new float[16] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] utt = new float[16] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] utotal = new float[16] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int T1 = 50;
    int T2 = 30;
    int tp0 = 0;
    public enum StyleB
    {
        walk = 0,
        run = 1,
        jump = 2
    }
    public enum StyleQ
    {
        trot = 0,
        bound = 1,
        pronk = 2
    }
    public enum StyleL
    {
        drive = 0,
        walk = 1,
        jump = 2
    }
    public enum StyleR
    {
        biped = 0,
        quadruped = 1,
        legwheeled = 2
    }

    Transform body;
    public int ObservationNum;
    public int ActionNum;
    [Header("RobotType")]
    public StyleR Robot;
    [Header("Biped")]
    public StyleB BipedTargetMotion;
    public ModelAsset BWalkPolicy;
    public ModelAsset BRunPolicy;
    public ModelAsset BJumpPolicy;
    [Header("Quadruped")]
    public StyleQ QuadrupedTargetMotion;
    public ModelAsset QTrotPolicy;
    public ModelAsset QBoundPolicy;
    public ModelAsset QPronkPolicy;
    [Header("Legwheeled")]
    public StyleL LegwheelTargetMotion;
    public ModelAsset LDrivePolicy;
    public ModelAsset LWalkPolicy;
    public ModelAsset LJumpPolicy;

    StyleB LastBmotion;
    StyleQ LastQmotion;
    StyleL LastLmotion;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Quaternion rot0;
    public ArticulationBody[] arts = new ArticulationBody[40];  // G2: 改为 public 供测试访问（arts[0] 是骨盆）
    public ArticulationBody[] acts = new ArticulationBody[16];  // M0: 临时改为 public 供测试访问
    GameObject robot;

    // D1: 按名称的关节索引映射（防止硬编码数字错误）
    [Header("关节索引映射")]
    [Tooltip("左膝索引（由 Initialize 自动设置）")]
    public int IdxLeftKnee = -1;

    [Tooltip("右膝索引（由 Initialize 自动设置）")]
    public int IdxRightKnee = -1;

    [Tooltip("左髋 roll 索引（由 Initialize 自动设置）")]
    public int IdxLeftHipRoll = -1;

    [Tooltip("右髋 roll 索引（由 Initialize 自动设置）")]
    public int IdxRightHipRoll = -1;

    [Tooltip("左踝 roll 索引（由 Initialize 自动设置）")]
    public int IdxLeftAnkleRoll = -1;

    [Tooltip("右踝 roll 索引（由 Initialize 自动设置）")]
    public int IdxRightAnkleRoll = -1;

    float[] kb = new float[16] { 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30 };
    float[] kb1 = new float[16] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float[] kb2 = new float[16] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    float dh = 25;
    float d0 = 15;
    float ko = 2;
    float kh = 0;

    public override void Initialize()
    {
        // AF1: 在 InitializeSensors() 之前读取 environment_parameters 并设置 VectorObservationSize
        // 访问 Academy.Instance 会触发延迟初始化并完成握手，此时会收到首批 environment_parameters
        var academy = Academy.Instance;
        var ep = academy.EnvironmentParameters;

        // AF1: 读取观测开关 (0.0=false, >= 0.5=true, -1=未设置保留默认)
        float envCoM = ep.GetWithDefault("enable_com_observation", -1f);
        float envPred = ep.GetWithDefault("enable_prediction_observation", -1f);
        float envFF = ep.GetWithDefault("enable_feedforward_control", -1f);

        string comSource = "scene_default";
        string predSource = "scene_default";
        string ffSource = "scene_default";

        if (envCoM >= 0f)
        {
            enableCoMObservation = envCoM >= 0.5f;
            comSource = "environment_parameters";
        }
        if (envPred >= 0f)
        {
            enablePredictionObservation = envPred >= 0.5f;
            predSource = "environment_parameters";
        }
        if (envFF >= 0f)
        {
            enableFeedforwardControl = envFF >= 0.5f;
            ffSource = "environment_parameters";
        }

        arts = this.GetComponentsInChildren<ArticulationBody>();
        ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if(arts[k].jointType.ToString() == "RevoluteJoint")
            {
                acts[ActionNum] = arts[k];
                print(acts[ActionNum]);
                ActionNum++;
            }
        }
        body = arts[0].GetComponent<Transform>();
        pos0 = body.position;
        rot0 = body.rotation;
        arts[0].GetJointPositions(P0);
        arts[0].GetJointVelocities(W0);

        // AF1: 计算并设置 VectorObservationSize（必须在 InitializeSensors 之前）
        int baseDim = 9 + 2 * ActionNum;
        int comDim = enableCoMObservation ? 5 : 0;
        int predDim = enablePredictionObservation ? 24 : 0;
        int calculatedDim = baseDim + comDim + predDim;

        var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        if (behaviorParams != null)
        {
            behaviorParams.BrainParameters.VectorObservationSize = calculatedDim;
        }

        // AG1/AG3: 打印生效的配置，放宽验证（environment_parameters 在 Initialize 时可能未下发）
        int trainingAreaCount = FindObjectsOfType<TrainingArea>().Length;
        float envRollAmplitude = ep.GetWithDefault("significant_roll_amplitude", -999f);

        Debug.Log($"========== [AE3] AC3 Training Configuration ==========");
        Debug.Log($"[AE3] TrainingArea count: {trainingAreaCount}");
        Debug.Log($"[AE3] Observation switches:");
        Debug.Log($"  - enableCoMObservation = {enableCoMObservation} (source: {comSource})");
        Debug.Log($"  - enablePredictionObservation = {enablePredictionObservation} (source: {predSource})");
        Debug.Log($"  - enableFeedforwardControl = {enableFeedforwardControl} (source: {ffSource})");
        Debug.Log($"[AE3] VectorObservationSize:");
        Debug.Log($"  - Calculated: {calculatedDim} (base={baseDim}, CoM={comDim}, Pred={predDim})");
        Debug.Log($"  - BehaviorParameters: {behaviorParams?.BrainParameters.VectorObservationSize ?? -1}");

        // AG3: significant_roll_amplitude 在 Initialize 时可能未下发，只打印不退出
        if (envRollAmplitude <= -999f)
        {
            Debug.LogWarning("[AG3] Wave amplitude not yet available in Initialize (will be set in first episode reset)");
        }
        else
        {
            Debug.Log($"[AE3] Wave amplitude: significant_roll_amplitude = {envRollAmplitude}°");
        }
        Debug.Log($"======================================================");

        // AE3: 配置验证（只验证结构性参数，运行时参数放宽）
        if (trainingAreaCount == 0)
        {
            Debug.LogError("[AE3] FATAL: TrainingArea count = 0, scene not configured!");
            #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
            #else
            Application.Quit(1);
            #endif
        }
        if (behaviorParams == null || behaviorParams.BrainParameters.VectorObservationSize != calculatedDim)
        {
            Debug.LogError($"[AE3] FATAL: VectorObservationSize mismatch! Expected {calculatedDim}, got {behaviorParams?.BrainParameters.VectorObservationSize ?? -1}");
            #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
            #else
            Application.Quit(1);
            #endif
        }

        // G2 修复：在父级 training area 中查找平台控制器（不用全局 Find）
        // Training area 架构：每个 agent 有独立的 ground，避免多 clone 共享平台
        Transform areaRoot = transform.parent;
        if (areaRoot != null)
        {
            platformController = areaRoot.GetComponentInChildren<WaveSpectrumPlatformController>();
            if (platformController != null)
            {
                Debug.Log($"[RobotRLAgent] {name} 在 area {areaRoot.name} 中找到独立平台控制器");
            }
            else
            {
                Debug.LogWarning($"[RobotRLAgent] {name} 未在 area {areaRoot.name} 中找到 WaveSpectrumPlatformController");
            }
        }
        else
        {
            // 兜底：无父级时尝试全局查找（单 agent 场景）
            GameObject ground = GameObject.Find("Ground");
            if (ground != null)
            {
                platformController = ground.GetComponent<WaveSpectrumPlatformController>();
                Debug.Log($"[RobotRLAgent] {name} 全局查找找到平台控制器");
            }
        }

        // 查找足部关节（用于接触检测）
        // 注意：这里通过名称查找，适用于 G1 等类似结构的机器人
        for (int i = 0; i < arts.Length; i++)
        {
            if (arts[i] != null)
            {
                if (arts[i].name.Contains("left_ankle_roll_link"))
                {
                    leftFootJoint = arts[i];
                    Debug.Log($"[RobotRLAgent] 找到左脚关节: {leftFootJoint.name}");
                }
                if (arts[i].name.Contains("right_ankle_roll_link"))
                {
                    rightFootJoint = arts[i];
                    Debug.Log($"[RobotRLAgent] 找到右脚关节: {rightFootJoint.name}");
                }
            }
        }

        if (leftFootJoint == null || rightFootJoint == null)
        {
            Debug.LogWarning($"[RobotRLAgent] 未找到足部关节！请检查关节命名");
        }

        // D1: 建立按名称的关节索引映射（防止硬编码索引错误）
        for (int i = 0; i < ActionNum; i++)
        {
            if (acts[i] == null) continue;
            string jointName = acts[i].name.ToLower();

            if (jointName.Contains("left") && jointName.Contains("knee"))
            {
                IdxLeftKnee = i;
                Debug.Log($"[RobotRLAgent] D1: 映射 IdxLeftKnee = {i} ({acts[i].name})");
            }
            else if (jointName.Contains("right") && jointName.Contains("knee"))
            {
                IdxRightKnee = i;
                Debug.Log($"[RobotRLAgent] D1: 映射 IdxRightKnee = {i} ({acts[i].name})");
            }
            else if (jointName.Contains("left") && jointName.Contains("hip") && jointName.Contains("roll"))
            {
                IdxLeftHipRoll = i;
                Debug.Log($"[RobotRLAgent] D1: 映射 IdxLeftHipRoll = {i} ({acts[i].name})");
            }
            else if (jointName.Contains("right") && jointName.Contains("hip") && jointName.Contains("roll"))
            {
                IdxRightHipRoll = i;
                Debug.Log($"[RobotRLAgent] D1: 映射 IdxRightHipRoll = {i} ({acts[i].name})");
            }
            else if (jointName.Contains("left") && jointName.Contains("ankle") && jointName.Contains("roll"))
            {
                IdxLeftAnkleRoll = i;
                Debug.Log($"[RobotRLAgent] D1: 映射 IdxLeftAnkleRoll = {i} ({acts[i].name})");
            }
            else if (jointName.Contains("right") && jointName.Contains("ankle") && jointName.Contains("roll"))
            {
                IdxRightAnkleRoll = i;
                Debug.Log($"[RobotRLAgent] D1: 映射 IdxRightAnkleRoll = {i} ({acts[i].name})");
            }
        }

        // D1: 验证所有关键关节已找到（膝关节必须，其他可选）
        if (IdxLeftKnee < 0 || IdxRightKnee < 0)
        {
            string errorMsg = $"[RobotRLAgent] D1 严重错误：未找到关键关节！" +
                              $"IdxLeftKnee={IdxLeftKnee}, IdxRightKnee={IdxRightKnee}。" +
                              $"前馈控制器无法工作，必须修复关节命名或映射逻辑。";
            Debug.LogError(errorMsg);
            throw new System.Exception(errorMsg);
        }

        Debug.Log($"[RobotRLAgent] D1 关节索引映射完成：" +
                  $"LeftKnee={IdxLeftKnee}, RightKnee={IdxRightKnee}, " +
                  $"LeftHipRoll={IdxLeftHipRoll}, RightHipRoll={IdxRightHipRoll}, " +
                  $"LeftAnkleRoll={IdxLeftAnkleRoll}, RightAnkleRoll={IdxRightAnkleRoll}");

        // 初始化显式速度预测器
        velocityPredictor = gameObject.GetComponent<ExplicitVelocityPredictor>();
        if (velocityPredictor == null)
        {
            velocityPredictor = gameObject.AddComponent<ExplicitVelocityPredictor>();
            Debug.Log($"[RobotRLAgent] 已添加 ExplicitVelocityPredictor 组件");
        }
        else
        {
            Debug.Log($"[RobotRLAgent] 已找到现有 ExplicitVelocityPredictor 组件");
        }

        // M1: 初始化甲板 IMU (真前馈传感器)
        deckIMU = gameObject.GetComponent<DeckIMU>();
        if (deckIMU == null)
        {
            deckIMU = gameObject.AddComponent<DeckIMU>();
            Debug.Log($"[RobotRLAgent] 已添加 DeckIMU 组件 (M1 真前馈)");
        }
        else
        {
            Debug.Log($"[RobotRLAgent] 已找到现有 DeckIMU 组件 (M1 真前馈)");
        }

        // 初始化预测前馈控制器
        predictiveController = gameObject.GetComponent<PredictiveFeedforwardController>();
        if (predictiveController == null)
        {
            predictiveController = gameObject.AddComponent<PredictiveFeedforwardController>();
            Debug.Log($"[RobotRLAgent] 已添加 PredictiveFeedforwardController 组件");
        }
        else
        {
            Debug.Log($"[RobotRLAgent] 已找到现有 PredictiveFeedforwardController 组件");
        }

        // P0-3 修复：应用配置系统参数
        ApplyExperimentConfig();

        // P0-4 修复：移到 Initialize() 并改为异常
        // 验证观测维度与 BehaviorParameters 配置一致
        ValidateObservationDimension();
    }

    /// <summary>
    /// T2: 获取期望的观测维度（用于填充默认观测）
    /// </summary>
    private int GetExpectedObservationSize()
    {
        int baseDim = 9 + 2 * ActionNum;
        int comDim = enableCoMObservation ? 5 : 0;
        int predDim = enablePredictionObservation ? 24 : 0;
        return baseDim + comDim + predDim;
    }

    /// <summary>
    /// P0-4 修复：验证观测维度，不匹配时抛异常
    /// 移到 Initialize() 确保训练开始前就发现问题
    /// </summary>
    private void ValidateObservationDimension()
    {
        // 计算实际观测维度的各个部分
        int baseDim = 9 + 2 * ActionNum;
        int comDim = enableCoMObservation ? 5 : 0;
        int predDim = enablePredictionObservation ? 24 : 0;
        int calculatedDim = baseDim + comDim + predDim;

        var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        if (behaviorParams == null)
        {
            Debug.LogWarning("[RobotRLAgent] 未找到 BehaviorParameters 组件，跳过维度验证");
            return;
        }

        int configuredSize = behaviorParams.BrainParameters.VectorObservationSize;

        if (calculatedDim != configuredSize)
        {
            string errorMsg = $"[RobotRLAgent] ❌ 观测维度不匹配！训练无法进行。\n" +
                              $"  计算维度: {calculatedDim}\n" +
                              $"    - 基础: {baseDim}\n" +
                              $"    - CoM: {comDim} (enableCoMObservation={enableCoMObservation})\n" +
                              $"    - Pred: {predDim} (enablePredictionObservation={enablePredictionObservation})\n" +
                              $"  配置维度 (BehaviorParameters.VectorObservationSize): {configuredSize}\n" +
                              $"请修改 Unity 场景中的 BehaviorParameters.VectorObservationSize 为 {calculatedDim}";

            Debug.LogError(errorMsg);
            throw new System.InvalidOperationException(errorMsg);
        }

        Debug.Log($"[RobotRLAgent] ✓ 观测维度验证通过: {calculatedDim}");
    }

    /// <summary>
    /// P0-3 修复：应用实验配置系统参数
    /// 从 ExperimentConfig 组件读取配置并应用到各个子组件
    /// </summary>
    private void ApplyExperimentConfig()
    {
        // 查找 ExperimentConfig 组件
        var experimentConfig = gameObject.GetComponent<ExperimentConfig>();
        if (experimentConfig == null)
        {
            Debug.Log("[RobotRLAgent] 未找到 ExperimentConfig 组件，使用 Inspector 默认值");
            return;
        }

        var config = experimentConfig.GetConfig();
        if (config == null)
        {
            Debug.LogWarning("[RobotRLAgent] ExperimentConfig 返回 null，跳过配置应用");
            return;
        }

        // L3: 应用观测配置并动态设置 VectorObservationSize
        enableCoMObservation = config.observation.enableCoMObservation;
        enablePredictionObservation = config.observation.enablePredictionObservation;
        enableFeedforwardControl = config.observation.enableFeedforwardControl;

        // 计算期望的观测维度
        int baseDim = 9 + 2 * ActionNum;
        int comDim = enableCoMObservation ? 5 : 0;
        int predDim = enablePredictionObservation ? 24 : 0;
        int expectedDim = baseDim + comDim + predDim;

        // 设置 BehaviorParameters.VectorObservationSize
        var behaviorParams = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
        if (behaviorParams != null)
        {
            behaviorParams.BrainParameters.VectorObservationSize = expectedDim;
            Debug.Log($"[RobotRLAgent] ✓ 观测配置已应用: " +
                     $"CoM={enableCoMObservation}, Pred={enablePredictionObservation}, FF={enableFeedforwardControl}, " +
                     $"VectorObservationSize={expectedDim}");
        }

        // 应用前馈控制参数到 PredictiveFeedforwardController
        if (predictiveController != null)
        {
            predictiveController.K_roll = config.feedforward.K_roll;
            predictiveController.max_roll_comp = config.feedforward.max_roll_comp;
            predictiveController.support_threshold = config.feedforward.support_threshold;
            predictiveController.filter_alpha = config.feedforward.filter_alpha;

            Debug.Log($"[RobotRLAgent] ✓ 前馈控制参数已应用: " +
                     $"K_roll={config.feedforward.K_roll}, " +
                     $"max_comp={config.feedforward.max_roll_comp}°, " +
                     $"support_th={config.feedforward.support_threshold}, " +
                     $"filter_alpha={config.feedforward.filter_alpha}");
        }

        // 应用预测器参数到 ExplicitVelocityPredictor
        if (velocityPredictor != null)
        {
            // 注意：predictor_order 和 prediction_horizon 需要在 ExplicitVelocityPredictor 中暴露
            // 这里先打印日志，实际应用在 ExplicitVelocityPredictor 中完成
            Debug.Log($"[RobotRLAgent] 预测器配置: " +
                     $"order={config.feedforward.predictor_order}, " +
                     $"horizon={config.feedforward.prediction_horizon}s");
        }

        // 应用平台参数（ExperimentConfig 已在 Awake() 中应用，这里无需重复）
        // 但可以记录日志确认
        if (platformController != null)
        {
            Debug.Log($"[RobotRLAgent] 平台运动配置: " +
                     $"roll={config.platform.rollAmplitude}°@{config.platform.rollFrequency}Hz, " +
                     $"pitch={config.platform.pitchAmplitude}°@{config.platform.pitchFrequency}Hz, " +
                     $"heave={config.platform.heaveAmplitude}m@{config.platform.heaveFrequency}Hz");
        }

        Debug.Log($"[RobotRLAgent] ✓ 实验配置应用完成: {config.meta.experimentName}");
    }


    private bool _isClone = false;
    void Start()
    {
        Time.fixedDeltaTime = 0.01f;

        /*SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset"));
        SerializedProperty layers = tagManager.FindProperty("layers");
        SerializedProperty layer = layers.GetArrayElementAtIndex(15);
        int targetLayer = LayerMask.NameToLayer("robot");
        layer.stringValue = "robot";
        tagManager.ApplyModifiedProperties();
        Physics.IgnoreLayerCollision(15, 15, true);
        ChangeLayerRecursively(gameObject, 15);*/

        // G2 修复：克隆整个 training area（robot + ground），每个 area 独立海况
        // X1(b): 增加 spawnClonesAtRuntime 开关，默认关闭（使用 --num-envs 多进程）
        if (train && spawnClonesAtRuntime && !_isClone)
        {
            Transform areaRoot = transform.parent;

            if (areaRoot != null && areaRoot.name != "Root" && areaRoot.GetComponent<Canvas>() == null)
            {
                // 有 training area 父级：克隆整个 area
                for (int i = 1; i < 14; i++)
                {
                    GameObject clonedArea = Instantiate(areaRoot.gameObject);
                    clonedArea.transform.position = areaRoot.position + new Vector3(i * 5f, 0, 0);  // 5m 间距
                    clonedArea.name = $"{areaRoot.name}_Clone_{i}";

                    // 标记 area 内的 agent 为 clone
                    RobotRLAgent clonedAgent = clonedArea.GetComponentInChildren<RobotRLAgent>();
                    if (clonedAgent != null)
                    {
                        clonedAgent._isClone = true;
                        Debug.Log($"[RobotRLAgent] 克隆 training area {i}: {clonedArea.name}");
                    }
                }
            }
            else
            {
                // 无 area 父级：退回旧逻辑，只克隆机器人（单平台模式）
                Debug.LogWarning("[RobotRLAgent] 未检测到 training area 结构，退回单平台克隆模式");
                for (int i = 1; i < 14; i++)
                {
                    GameObject clone = Instantiate(gameObject);
                    clone.transform.position = transform.position + new Vector3(i * 2f, 0, 0);
                    clone.name = $"{name}_Clone_{i}";
                    clone.GetComponent<RobotRLAgent>()._isClone = true;
                }
            }
        }
    }
    void ChangeLayerRecursively(GameObject obj, int targetLayer)
    {
        obj.layer = targetLayer;
        foreach (Transform child in obj.transform)ChangeLayerRecursively(child.gameObject, targetLayer);
    }

    public override void OnEpisodeBegin()
    {
        // AF1: OnEpisodeBegin 不再读取观测开关（已在 Initialize 中完成）
        // 这里只处理可能每个 episode 变化的参数（如课程学习的波浪幅度）

        // C: 输出上一 episode 的 posture RMSE（在重置前）
        if (postureStepCount > 0)
        {
            float rollRMSE = Mathf.Sqrt(rollErrorSumSquared / postureStepCount);
            float pitchRMSE = Mathf.Sqrt(pitchErrorSumSquared / postureStepCount);
            Debug.Log($"[RobotRLAgent] Episode Posture RMSE - Roll: {rollRMSE:F3}°, Pitch: {pitchRMSE:F3}°, Steps: {postureStepCount}");
        }

        // C: 重置 posture RMSE 累积
        rollErrorSumSquared = 0f;
        pitchErrorSumSquared = 0f;
        postureStepCount = 0;

        // T2: 报告上一episode的错误统计
        if (observationErrorCount > 0)
        {
            Debug.LogWarning($"[RobotRLAgent] 上一episode观测错误次数: {observationErrorCount}");
        }

        // T2: 重置错误计数和销毁标志
        observationErrorCount = 0;
        isDestroying = false;

        tp = 0;
        tq = 0;
        tt = 0;
        for (int i = 0; i< 16; i++) u[i] = 0;
        for (int i = 0; i < 16; i++) ut[i] = 0;
        for (int i = 0; i < 16; i++) utt[i] = 0;

        // P0-4 修复：根据开关动态计算观测维度
        // 基础观察：9 + 2 * ActionNum
        int baseDim = 9 + 2 * ActionNum;  // 通常 = 9 + 24 = 33

        // CoM + 足部接触观测：3 + 1 + 1 = 5
        int comDim = enableCoMObservation ? 5 : 0;

        // 显式速度预测观测：
        //   - 速度导数(3) + 速度二阶导(3) + 预测速度(3) = 9
        //   - 角速度导数(3) + 角速度二阶导(3) + 预测角速度(3) = 9
        //   - 当前欧拉角(3) + 预测欧拉角(3) = 6
        //   小计：24
        int predDim = enablePredictionObservation ? 24 : 0;

        ObservationNum = baseDim + comDim + predDim;

        // 消融配置对应的维度：
        // Baseline (33):      enableCoMObservation=false, enablePredictionObservation=false → 33 + 0 + 0 = 33
        // +CoM (38):          enableCoMObservation=true,  enablePredictionObservation=false → 33 + 5 + 0 = 38
        // +Pred (57):         enableCoMObservation=false, enablePredictionObservation=true  → 33 + 0 + 24 = 57
        // +CoM+Pred (62):     enableCoMObservation=true,  enablePredictionObservation=true  → 33 + 5 + 24 = 62

        // P0-4: 维度验证已移至 Initialize()，此处无需重复验证

        // H4: ArticulationBody 引用检查 - 异常情况抛错，不静默跳过
        if (arts == null || arts.Length == 0 || arts[0] == null)
        {
            Debug.LogError("[RobotRLAgent] ArticulationBody 引用无效 - 这是严重错误，会导致 episode 静默失败");
            throw new System.NullReferenceException(
                "ArticulationBody 引用在 OnEpisodeBegin 时为 null。" +
                "这会导致机器人未重置到初始位姿，产生垃圾 episode。" +
                "如果在测试中遇到此问题，应在测试的 TearDown 中正确清理，而非在生产代码中 workaround。");
        }

        if (fixbody) arts[0].immovable = true;
        if (!fixbody)
        {
            arts[0].TeleportRoot(pos0, rot0);
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
            arts[0].SetJointPositions(P0);
            arts[0].SetJointVelocities(W0);
        }

        // G2: Training Area 架构 - 每个 agent 重采样并重置自己的平台
        // 每个 area 拥有独立平台，可以安全地在 episode 边界重采样海况
        if (platformController != null)
        {
            platformController.ResampleWave();  // 随机化海况参数 (Hs, Tp, 相位)
            platformController.ResetMotion();   // 重置到初始姿态
        }

        // 重置预测控制器的滤波器
        if (predictiveController != null)
        {
            predictiveController.ResetFilter();
        }
    }

    /// <summary>
    /// T2: 场景销毁时设置标志，防止CollectObservations在销毁时被调用
    /// </summary>
    protected override void OnDisable()
    {
        isDestroying = true;

        // 报告最终错误统计
        if (observationErrorCount > 0)
        {
            Debug.LogWarning($"[RobotRLAgent] OnDisable - 总观测错误次数: {observationErrorCount}");
        }
    }

    /// <summary>
    /// C: 碰撞回调，记录足部接触力
    /// 每个 FixedUpdate 前清零，在碰撞时累加
    /// </summary>
    void OnCollisionStay(Collision collision)
    {
        // 检查是否是足部碰撞
        ArticulationBody collidedBody = collision.articulationBody;
        if (collidedBody == null) return;

        // C: 计算法向接触力：使用 Collision.impulse 估计
        // Unity's Collision.impulse 是总冲量向量（kg*m/s）
        // impulse.magnitude / Time.fixedDeltaTime 约等于平均接触力（N）
        float normalForce = collision.impulse.magnitude / Time.fixedDeltaTime;

        // 根据碰撞体判断是左脚还是右脚
        if (collidedBody == leftFootJoint)
        {
            leftFootContactForce = Mathf.Max(leftFootContactForce, normalForce);
        }
        else if (collidedBody == rightFootJoint)
        {
            rightFootContactForce = Mathf.Max(rightFootContactForce, normalForce);
        }
    }

    /// <summary>
    /// C: 检测足部是否接触地面（改进版）
    /// 主判据：接触力 > 阈值（体重的5%）
    /// 备用判据：相对平台表面高度 < 阈值（用于力传感器失效时）
    /// </summary>
    private bool IsFootInContact(ArticulationBody footJoint)
    {
        if (footJoint == null) return false;

        // 主判据：接触力
        float contactForce = 0f;
        if (footJoint == leftFootJoint)
        {
            contactForce = leftFootContactForce;
        }
        else if (footJoint == rightFootJoint)
        {
            contactForce = rightFootContactForce;
        }

        if (contactForce > contactForceThreshold)
        {
            return true;
        }

        // 备用判据：相对平台表面的高度（而非世界高度）
        // 在倾斜甲板上，世界高度与实际地面距离不一致
        if (platformController != null)
        {
            Transform platformTransform = platformController.transform;
            Vector3 footLocalPos = platformTransform.InverseTransformPoint(footJoint.transform.position);

            // footLocalPos.y 是足部相对平台局部坐标的 Y（垂直于平台表面）
            // 阈值 contactHeightThreshold 表示距平台表面的最大高度
            if (footLocalPos.y < contactHeightThreshold)
            {
                return true;
            }
        }
        else
        {
            // 如果没有平台控制器，退化为世界高度判据
            float footHeight = footJoint.transform.position.y;
            if (footHeight < contactHeightThreshold)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// C: 在每个 FixedUpdate 开始时清零接触力缓存
    /// </summary>
    private void ResetContactForces()
    {
        leftFootContactForce = 0f;
        rightFootContactForce = 0f;
    }

    /// <summary>
    /// T3: 计算全身合质心（正确的方法）
    /// P0-1 修复：替代原来只用 arts[0].worldCenterOfMass 的错误方法
    /// </summary>
    public Vector3 GetWholeBodyCoM()
    {
        float totalMass = 0f;
        Vector3 weightedSum = Vector3.zero;

        foreach (var ab in arts)
        {
            if (ab == null) continue;

            // 逐 link 的质心 × 质量
            weightedSum += ab.worldCenterOfMass * ab.mass;
            totalMass += ab.mass;
        }

        if (totalMass <= 0f)
        {
            Debug.LogError("[RobotRLAgent] 计算质心时总质量为0！");
            return Vector3.zero;
        }

        return weightedSum / totalMass;
    }

    /// <summary>
    /// P0-2 修复：获取上一步实际施加的前馈补偿值
    /// T3: 返回副本以避免外部修改
    /// 注意：这是实际叠加到关节控制的值，包含相位门控和饱和限制
    /// </summary>
    public float[] GetLastPredictiveCompensation()
    {
        float[] copy = new float[lastPredictiveComp.Length];
        System.Array.Copy(lastPredictiveComp, copy, lastPredictiveComp.Length);
        return copy;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // T2: 场景销毁时跳过（OnDisable已标记）
        if (isDestroying) {
            // 填充默认观测以保持维度一致
            int expectedDim = GetExpectedObservationSize();
            for (int i = 0; i < expectedDim; i++) {
                sensor.AddObservation(0f);
            }
            return;
        }

        // T2: 防御性检查（记录错误但不return，避免0维观测）
        if (acts == null || arts == null || body == null || ActionNum > acts.Length)
        {
            Debug.LogError($"[RobotRLAgent] CollectObservations error: acts/arts/body null or ActionNum invalid");
            observationErrorCount++;
            // 填充默认观测
            int expectedDim = GetExpectedObservationSize();
            for (int i = 0; i < expectedDim; i++) {
                sensor.AddObservation(0f);
            }
            return;
        }

        // 基础观察值 (33 维) - 始终添加
        sensor.AddObservation(body.InverseTransformDirection(Vector3.down));  // 3
        sensor.AddObservation(body.InverseTransformDirection(arts[0].angularVelocity));  // 3
        sensor.AddObservation(body.InverseTransformDirection(arts[0].velocity));  // 3
        for (int i = 0; i < ActionNum; i++)  // 24 (12 joints * 2)
        {
            // T2: 检查但不early-return
            if (acts[i] == null || acts[i].jointPosition.dofCount == 0) {
                Debug.LogError($"[RobotRLAgent] Joint {i} null or invalid DOF");
                observationErrorCount++;
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            } else {
                sensor.AddObservation(acts[i].jointPosition[0]);
                sensor.AddObservation(acts[i].jointVelocity[0]);
            }
        }

        // CoM观察值 (5 维) - 根据enableCoMObservation开关添加
        if (enableCoMObservation)
        {
            // 1. CoM (质心) 相对于躯干的位置 (3 维)
            // P0-1 修复：使用全身合质心而非单个 link 的质心
            Vector3 worldCoM = GetWholeBodyCoM();
            Vector3 comLocal = body.InverseTransformPoint(worldCoM);
            sensor.AddObservation(comLocal);  // 3

            // 2. 左脚接触状态 (1 维)
            float leftContact = IsFootInContact(leftFootJoint) ? 1.0f : 0.0f;
            sensor.AddObservation(leftContact);  // 1

            // 3. 右脚接触状态 (1 维)
            float rightContact = IsFootInContact(rightFootJoint) ? 1.0f : 0.0f;
            sensor.AddObservation(rightContact);  // 1
        }

        // 预测观察值 (24 维) - 根据enablePredictionObservation开关添加
        if (enablePredictionObservation && velocityPredictor != null)
        {
            // 获取预测器的简化观测（仅包含预测值，不包含所有导数）
            Vector3 predVel, predAngVel, predEuler;
            velocityPredictor.GetSimplifiedPredictionObservations(out predVel, out predAngVel, out predEuler);

            // 4.1 线速度预测相关 (9 维)
            sensor.AddObservation(velocityPredictor.velocityFirstDerivative);     // 一阶导数（加速度）3 维
            sensor.AddObservation(velocityPredictor.velocitySecondDerivative);    // 二阶导数（jerk）3 维
            sensor.AddObservation(predVel);                                        // 预测速度 3 维

            // 4.2 角速度预测相关 (9 维)
            sensor.AddObservation(velocityPredictor.angularVelocityFirstDerivative);  // 角加速度 3 维
            sensor.AddObservation(velocityPredictor.angularVelocitySecondDerivative); // 角jerk 3 维
            sensor.AddObservation(predAngVel);                                         // 预测角速度 3 维

            // 4.3 姿态预测相关 (6 维)
            // P0-5 修复：欧拉角归一化到 [-180, 180] 避免过零点跳变
            Vector3 currentEuler = body.eulerAngles;
            sensor.AddObservation(EulerTrans(currentEuler.x));  // pitch
            sensor.AddObservation(EulerTrans(currentEuler.y));  // yaw
            sensor.AddObservation(EulerTrans(currentEuler.z));  // roll

            sensor.AddObservation(EulerTrans(predEuler.x));     // 预测 pitch
            sensor.AddObservation(EulerTrans(predEuler.y));     // 预测 yaw
            sensor.AddObservation(EulerTrans(predEuler.z));     // 预测 roll
        }

        // 观测维度会根据开关自动调整：
        // Baseline (33-dim):        enableCoMObservation=false, enablePredictionObservation=false
        // +CoM (38-dim):            enableCoMObservation=true,  enablePredictionObservation=false
        // +Pred (57-dim):           enableCoMObservation=false, enablePredictionObservation=true
        // +CoM+Pred (62-dim):       enableCoMObservation=true,  enablePredictionObservation=true
        // Ours (62-dim with FF):    enableCoMObservation=true,  enablePredictionObservation=true, enableFeedforwardControl=true
    }
    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180)
            return eulerAngle;
        else
            return eulerAngle - 360f;
    }
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        for (int i = 0; i < 16; i++) utotal[i] = 0;
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;

        for (int i = 0; i < ActionNum; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            ut[i] += u[i];
            utt[i] += ut[i];
            utotal[i] = kb[i] * u[i] + kb1[i] * ut[i] + kb2[i] * utt[i];
            if (fixbody) utotal[i] = 0;
        }

        string name = this.name;
        int[] idx = new int[6] { 2, 3, 4, 7, 8, 9 };
        float[] ktemp1 = new float[12] { 5, 5, 30, 60, 30, 5, 5, 30, 60, 30, 0, 0 };
        d0 = 0;
        if (name.Contains("Tinker"))
        {
            idx = new int[6] { 2, 3, 4, 7, 8, 9 };
            ktemp1 = new float[12] { 5, 5, 30, 60, 30, 5, 5, 30, 60, 30, 0, 0 };
            ko = 01f;
            d0 = 20;
        }
        if (name.Contains("TaiTan"))
        {
            idx = new int[6] { 2, 3, 4, -7, -8, 9 };
            ktemp1 = new float[12] { 5, 5, 30, 60, 30, 5, 5, 30, 60, 30, 0, 0 };
            ko = 1f;
        }
        if (name.Contains("Birdy"))
        {
            idx = new int[6] { 2, 3, 4, 7, 8, 9 };
            ktemp1 = new float[12] { 5, 5, 30, 60, 30, 5, 5, 30, 60, 30, 0, 0 };
            ko = 01f;
            d0 = 40;
        }
        if (Robot == StyleR.biped && ActionNum == 10)
        {
            if (BipedTargetMotion == StyleB.walk)
            {
                for (int i = 0; i < 12; i++) kb[i] = ktemp1[i];
                T1 = 30;
                dh = 20;
                utotal[Mathf.Abs(idx[0])] += (dh * uf1 + d0) * Mathf.Sign(idx[0]);
                utotal[Mathf.Abs(idx[1])] -= 2 * (dh * uf1 + d0) * Mathf.Sign(idx[1]);
                utotal[Mathf.Abs(idx[2])] += (dh * uf1 + d0) * Mathf.Sign(idx[2]);
                utotal[Mathf.Abs(idx[3])] += (dh * uf2 + d0) * Mathf.Sign(idx[3]);
                utotal[Mathf.Abs(idx[4])] -= 2 * (dh * uf2 + d0) * Mathf.Sign(idx[4]);
                utotal[Mathf.Abs(idx[5])] += (dh * uf2 + d0) * Mathf.Sign(idx[5]);
                if (!train) SetModel("gewu", BWalkPolicy);
            }
            if (BipedTargetMotion == StyleB.run)
            {
                float[] ktemp = new float[12] { 5, 9, 40, 60, 40, 5, 9, 40, 60, 40, 0, 0 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                T1 = 20;
                dh = 20;
                d0 = 15;
                utotal[2] += (dh * uf1 + d0);
                utotal[3] -= 2 * (dh * uf1 + d0);
                utotal[4] += (dh * uf1 + d0);
                utotal[7] += (dh * uf2 + d0);
                utotal[8] -= 2 * (dh * uf2 + d0);
                utotal[9] += (dh * uf2 + d0);
                if (!train) SetModel("gewu", BRunPolicy);
                if (tt > 900 && kh == 0)
                {
                    kh = 2;
                    ko = 4;
                    print(222222222222222);
                }
            }
            if (BipedTargetMotion == StyleB.jump)
            {
                float[] ktemp = new float[12] { 5, 5, 40, 60, 40, 5, 5, 40, 60, 40, 0, 0 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                T1 = 30;
                dh = 40;
                d0 = 5;
                utotal[2] += dh * uff + d0;
                utotal[3] -= (dh * uff + d0) * 2;
                utotal[4] += dh * uff + d0;
                utotal[7] = utotal[2];
                utotal[8] = utotal[3];
                utotal[9] = utotal[4];
                //utotal[7] += dh * uff + d0;
                //utotal[8] -= (dh * uff + d0) * 2;
                //utotal[9] += dh * uff + d0;
                if (!train) SetModel("gewu", BJumpPolicy);
                if (tt > 900 && kh == 0)
                {
                    kh = 2;
                    print(222222222222222);
                }
            }
        }

        idx = new int[6] { 3, 4, 5, 9, 10, 11 };
        ktemp1 = new float[12]{ 10, 10, 45, 20, 40, 10,   10, 10, 40, 20, 45, 10 };
        dh = 10;
        d0 = 0;
        if (name.Contains("H1"))
        {
            idx = new int[6] { -2, -4, -5, -8, -10, -11 };
            ktemp1 = new float[12] { 10, 30, 10, 30, 30, 10,    10, 30, 10, 30, 30, 10 };
            ko = 01f;
            d0 = 10;
            dh = 10;
        }
        if (name.Contains("G1"))
        {
            idx = new int[6] { -1, -4, -5, -7, -10, -11 };
            ktemp1 = new float[12] { 40, 10, 10, 20, 40, 10, 40, 10, 10, 20, 40, 10 };
            ko = 01f;
            d0 = 10;
            dh = 30;
        }
        if (name.Contains("Mlg"))
        {
            idx = new int[6] { 3, 4, 5, -9, -10, -11 };
            ktemp1 = new float[12] { 40, 10, 10, 20, 40, 10, 40, 10, 10, 20, 40, 10 };
            dh = 30;
        }
        if (name.Contains("MiniLoong"))
        {
            idx = new int[6] { -1, 4, -5, 7, -10, -11 };
            ktemp1 = new float[12] { 40, 30, 30, 30, 40, 30, 40, 30, 30, 30, 40, 30 };
            ko = 01f;
            dh = 30;
        }
        if (name.Contains("V2.5"))
        {
            idx = new int[6] { -3, -4, -5, -9, -10, -11 };
            ktemp1 = new float[12] { 40, 10, 10, 20, 40, 10, 40, 10, 10, 20, 40, 10 };
            ko = 0.1f;
            dh = 30;
        }
        if (name.Contains("T1"))
        {
            idx = new int[6] { -1, -4, -5, -7, -10, -11 };
            ktemp1 = new float[12] { 40, 10, 10, 20, 40, 10, 40, 10, 10, 20, 40, 10 };
            ko = 01f;
            d0 = 5;
            dh = 30;
        }
        if (name.Contains("zqsa01"))
        {
            idx = new int[6] { -3, -4, -5, -9, -10, -11 };
            ktemp1 = new float[12] { 40, 10, 10, 20, 40, 10, 40, 10, 10, 20, 40, 10 };
            ko = 01f;
            d0 = 0;
            dh = 40;
        }
        if (Robot == StyleR.biped && ActionNum == 12)
        {
            if (BipedTargetMotion == StyleB.walk)
            {
                for (int i = 0; i < 12; i++) kb[i] = ktemp1[i];
                T1 = 40;
                utotal[Mathf.Abs(idx[0]) - 1] += (dh * uf1 + d0) * Mathf.Sign(idx[0]);
                utotal[Mathf.Abs(idx[1]) - 1] -= 2 * (dh * uf1 + d0) * Mathf.Sign(idx[1]);
                utotal[Mathf.Abs(idx[2]) - 1] += (dh * uf1 + d0) * Mathf.Sign(idx[2]);
                utotal[Mathf.Abs(idx[3]) - 1] += (dh * uf2 + d0) * Mathf.Sign(idx[3]);
                utotal[Mathf.Abs(idx[4]) - 1] -= 2 * (dh * uf2 + d0) * Mathf.Sign(idx[4]);
                utotal[Mathf.Abs(idx[5]) - 1] += (dh * uf2 + d0) * Mathf.Sign(idx[5]);
                if (!train) SetModel("gewu", BWalkPolicy);
            }
            if (BipedTargetMotion == StyleB.run)
            {
                float[] ktemp = new float[12] { 10, 10, 60, 30, 60, 10,    10, 10, 60, 30, 60, 10 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                T1 = 25;
                dh = 40;
                d0 = 20;
                utotal[2] += (dh * uf1 + d0);
                utotal[3] -= 2 * (dh * uf1 + d0);
                utotal[4] += (dh * uf1 + d0);
                utotal[8] += (dh * uf2 + d0);
                utotal[9] -= 2 * (dh * uf2 + d0);
                utotal[10] += (dh * uf2 + d0);
                if (!train) SetModel("gewu", BRunPolicy);
                if (tt > 900 && kh == 0)
                {
                    kh = 2;
                    //ko = 4;
                    print(222222222222222);
                }
            }
            if (BipedTargetMotion == StyleB.jump)
            {
                float[] ktemp = new float[12] { 10, 10, 40, 50, 40, 10,   10, 10, 40, 50, 40, 10 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                T1 = 30;
                dh = 50;
                d0 = 5;
                utotal[2] += dh * uff + d0;
                utotal[3] -= (dh * uff + d0) * 2;
                utotal[4] += dh * uff + d0;
                //utotal[8] = utotal[0];
                //utotal[9] = utotal[3];
                //utotal[10] = utotal[4];
                utotal[8] += dh * uff + d0;
                utotal[9] -= (dh * uff + d0) * 2;
                utotal[10] += dh * uff + d0;
                if (!train) SetModel("gewu", BJumpPolicy);
                if (tt > 900 && kh == 0)
                {
                    kh = 4;
                    print(222222222222222);
                }
            }
        }
        if (Robot == StyleR.quadruped)
        {
            d0 = 30;
            dh = 15;
            ko = 1;
            T1 = 30;
            if (QuadrupedTargetMotion == StyleQ.trot)
            {
                utotal[1] += dh * uf1 + d0;
                utotal[2] += (dh * uf1 + d0) * -2;
                utotal[4] += dh * uf2 + d0;
                utotal[5] += (dh * uf2 + d0) * -2;
                utotal[7] += dh * uf2 + d0;
                utotal[8] += (dh * uf2 + d0) * -2;
                utotal[10] += dh * uf1 + d0;
                utotal[11] += (dh * uf1 + d0) * -2;
                if (!train) SetModel("gewu", QTrotPolicy);
            }
            if (QuadrupedTargetMotion == StyleQ.bound)
            {
                utotal[1] += dh * uf1 + d0;
                utotal[2] += (dh * uf1 + d0) * -2;
                utotal[4] += dh * uf1 + d0;
                utotal[5] += (dh * uf1 + d0) * -2;
                utotal[7] += dh * uf2 + d0;
                utotal[8] += (dh * uf2 + d0) * -2;
                utotal[10] += dh * uf2 + d0;
                utotal[11] += (dh * uf2 + d0) * -2;
                if (!train) SetModel("gewu", QBoundPolicy);
            }
            if (QuadrupedTargetMotion == StyleQ.pronk)
            {
                float[] ktemp = new float[12] { 5, 30, 60, 5, 30, 60, 5, 30, 60, 5, 30, 60 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                T2 = 30;
                dh = 40;
                d0 = 30;
                ko = 2;
                kh = 1;
                utotal[1] += dh * uff + d0;
                utotal[2] += (dh * uff + d0) * -2;
                utotal[4] += dh * uff + d0;
                utotal[5] += (dh * uff + d0) * -2;
                utotal[7] += dh * uff + d0;
                utotal[8] += (dh * uff + d0) * -2;
                utotal[10] += dh * uff + d0;
                utotal[11] += (dh * uff + d0) * -2;
                if (!train) SetModel("gewu", QPronkPolicy);
            }
        }
        if (Robot == StyleR.legwheeled  && ActionNum == 8)
        {

            if (LegwheelTargetMotion == StyleL.walk)
            {
                float[] ktemp = new float[12] { 10, 30, 30, 0, 10, 30, 30, 0, 0, 0, 0, 0 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                for (int i = 0; i < 12; i++) kb2[i] = 0;
                dh = 20;
                T1 = 30;
                ko = 1;
                utotal[1] -= dh * uf1;
                utotal[2] += dh * uf1 * -2;
                utotal[5] += dh * uf2;
                utotal[6] -= dh * uf2 * -2;
                if (!train) SetModel("gewu", LWalkPolicy);
            }

            if (LegwheelTargetMotion == StyleL.jump)
            {
                float[] ktemp = new float[12] { 10, 30, 30, 0, 10, 30, 30, 0, 0, 0, 0, 0 };
                for (int i = 0; i < 12; i++) kb[i] = ktemp[i];
                for (int i = 0; i < 12; i++) kb2[i] = 0;
                T2 = 40;
                kh = 5;
                utotal[1] -= 40 * uff;
                utotal[2] -= 80 * uff;
                utotal[5] += 40 * uff;
                utotal[6] += 80 * uff;
                if (!train) SetModel("gewu", LJumpPolicy);
            }
            if (LegwheelTargetMotion == StyleL.drive)
            {
                for (int i = 0; i < 12; i++) kb[i] = 0;
                for (int i = 0; i < 12; i++) kb2[i] = 0;
                kb2[3] = 0.2f;
                kb2[7] = 0.2f;
                //kb1[3] = 1f;
                //kb1[7] = 1f;
                if (!train) SetModel("gewu", LDrivePolicy);
            }
        }

        if (Robot == StyleR.legwheeled && ActionNum == 16)
        {
            if (LegwheelTargetMotion == StyleL.walk)
            {
                T1 = 30;
                dh = 20;
                d0 = 30;
                ko = 1;
                float[] ktemp = new float[16] { 10, 30, 30, 0, 10, 30, 30, 0, 10, 30, 30, 0, 10, 30, 30, 0 };
                for (int i = 0; i < 16; i++) kb[i] = ktemp[i];
                for (int i = 0; i < 16; i++) kb2[i] = 0;
                utotal[1] += dh * uf1 + d0;
                utotal[2] += (dh * uf1 + d0) * -2;
                utotal[5] += dh * uf2 + d0;
                utotal[6] += (dh * uf2 + d0) * -2;
                utotal[9] += dh * uf2 + d0;
                utotal[10] += (dh * uf2 + d0) * -2;
                utotal[13] += dh * uf1 + d0;
                utotal[14] += (dh * uf1 + d0) * -2;
                if (!train) SetModel("gewu", LWalkPolicy);
            }

            if (LegwheelTargetMotion == StyleL.jump)
            {
                T2 = 30;
                dh = 30;
                d0 = 30;
                kh = 2;
                float[] ktemp = new float[16] { 10, 30, 50, 0, 10, 30, 50, 0, 10, 30, 50, 0, 10, 30, 50, 0 };
                for (int i = 0; i < 16; i++) kb[i] = ktemp[i];
                for (int i = 0; i < 16; i++) kb2[i] = 0;
                utotal[1] += dh * uff + d0;
                utotal[2] += (dh * uff + d0) * -2;
                utotal[5] += dh * uff + d0;
                utotal[6] += (dh * uff + d0) * -2;
                utotal[9] += dh * uff + d0;
                utotal[10] += (dh * uff + d0) * -2;
                utotal[13] += dh * uff + d0;
                utotal[14] += (dh * uff + d0) * -2;
                if (!train) SetModel("gewu", LJumpPolicy);
            }
            if (LegwheelTargetMotion == StyleL.drive)
            {
                d0 = 30;
                utotal[1] += d0;
                utotal[2] += d0 * -2;
                utotal[5] += d0;
                utotal[6] += d0 * -2;
                utotal[9] += d0;
                utotal[10] += d0 * -2;
                utotal[13] += d0;
                utotal[14] += d0 * -2;
                //kb1[3] = 1f;
                //kb1[7] = 1f;
                for (int i = 0; i < 16; i++) kb[i] = 0;
                for (int i = 0; i < 16; i++) kb2[i] = 0;
                kb2[3] = 01f;
                kb2[7] = 01f;
                kb2[11] = 01f;
                kb2[15] = 01f;
                if (!train) SetModel("gewu", LDrivePolicy);
            }
        }

        // ========== 应用预测前馈补偿 ==========
        // 在gewu步态之后、应用到关节之前插入
        // 根据enableFeedforwardControl开关决定是否启用前馈控制
        if (enableFeedforwardControl &&
            predictiveController != null &&
            velocityPredictor != null)
        {
            // 获取预测的姿态
            Vector3 predVel, predAngVel, predEuler;
            velocityPredictor.GetSimplifiedPredictionObservations(out predVel, out predAngVel, out predEuler);

            // 计算预测补偿
            float[] predictiveComp = predictiveController.ComputeKneeCompensation(
                predEuler,
                uf1,
                uf2,
                ActionNum
            );

            // P0-2 修复：缓存实际施加的补偿值（用于日志记录）
            for (int i = 0; i < ActionNum; i++)
            {
                lastPredictiveComp[i] = predictiveComp[i];
            }

            // 叠加到utotal
            for (int i = 0; i < ActionNum; i++)
            {
                utotal[i] += predictiveComp[i];
            }
        }
        else
        {
            // P0-2：如果未启用前馈，补偿值为 0
            for (int i = 0; i < ActionNum; i++)
            {
                lastPredictiveComp[i] = 0f;
            }
        }

        for (int i = 0; i < ActionNum; i++) SetJointTargetDeg(acts[i], utotal[i]);


        if (BipedTargetMotion != LastBmotion || QuadrupedTargetMotion != LastQmotion || LegwheelTargetMotion != LastLmotion) EndEpisode();
        LastBmotion = BipedTargetMotion;
        LastQmotion = QuadrupedTargetMotion;
        LastLmotion = LegwheelTargetMotion;
    }
    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 2000f;
        drive.damping = 100f;
        drive.forceLimit = 300f;
        drive.target = x;
        joint.xDrive = drive;
    }
    void SetJointTargetPosition(ArticulationBody joint, float x)
    {
        x = (x + 1f) * 0.5f;
        var x1 = Mathf.Lerp(joint.xDrive.lowerLimit, joint.xDrive.upperLimit, x);
        var drive = joint.xDrive;
        drive.stiffness = 2000f;
        drive.damping = 100f;
        drive.forceLimit = 200f;
        drive.target = x1;
        joint.xDrive = drive;
    }
    public override void Heuristic(in ActionBuffers actionsOut)
    {

    }

    void FixedUpdate()
    {
        // C: 重置接触力缓存（每个物理步开始时）
        ResetContactForces();

        // M1: 更新速度预测器的历史状态
        // 使用甲板 IMU 数据（真前馈）而非机器人自身状态（超前校正）
        Vector3 platformVel, platformAngVel, platformPos, platformEuler;
        deckIMU.GetPlatformState(out platformVel, out platformAngVel, out platformPos, out platformEuler);
        velocityPredictor.UpdateHistory(
            platformVel,
            platformAngVel,
            platformPos,
            platformEuler
        );

        // C: 累积 posture 误差（用于 RMSE 计算）
        float roll = EulerTrans(body.eulerAngles[2]);  // z 轴是 roll
        float pitch = EulerTrans(body.eulerAngles[0]); // x 轴是 pitch
        rollErrorSumSquared += roll * roll;
        pitchErrorSumSquared += pitch * pitch;
        postureStepCount++;

        tp++;
        tq++;
        tt++;
        if (tp > 0 && tp <= T1)
        {
            tp0 = tp;
            uf1 = (-Mathf.Cos(3.14f * 2 * tp0 / T1) + 1f) / 2f;
            uf2 = 0;
        }
        if (tp > T1 && tp <= 2 * T1)
        {
            tp0 = tp - T1;
            uf1 = 0;
            uf2 = (-Mathf.Cos(3.14f * 2 * tp0 / T1) + 1f) / 2f;
        }
        if (tp >= 2 * T1) tp = 0;
        uff = (-Mathf.Cos(3.14f * 2 * tq / T2) + 1f) / 2f;
        if (tq >= T2) tq = 0;

        var vel = body.InverseTransformDirection(arts[0].velocity);
        var wel = body.InverseTransformDirection(arts[0].angularVelocity);
        var live_reward = 1f;
        var ori_reward1 = -0.1f * Mathf.Abs(EulerTrans(body.eulerAngles[0]));//-0.5f * Mathf.Min(Mathf.Abs(body.eulerAngles[0]), Mathf.Abs(body.eulerAngles[0] - 360f));
        var ori_reward2 = -2f * Mathf.Abs(wel[1]);
        var ori_reward3 = -0.1f * Mathf.Min(Mathf.Abs(body.eulerAngles[2]), Mathf.Abs(body.eulerAngles[2] - 360f));
        var vel_reward1 = vel[2] - Mathf.Abs(vel[0]);
        var vel_reward2 = vel[2] - Mathf.Abs(vel[0]) + kh * Mathf.Abs(vel[1]);

        // CoM 稳定性奖励：惩罚质心偏离躯干中心
        // comOffset = 质心在 xz 平面上偏离躯干中心的距离
        // P0-1 修复：使用全身合质心而非单个 link 的质心
        Vector3 worldCoM = GetWholeBodyCoM();
        Vector3 comLocal = body.InverseTransformPoint(worldCoM);
        float comOffset = Mathf.Sqrt(comLocal.x * comLocal.x + comLocal.z * comLocal.z);
        var stability_reward = -1.0f * comOffset;

        // 总奖励 = 存活奖励 + 姿态惩罚 + 速度奖励 + 稳定性奖励
        var reward = live_reward + (ori_reward1 + ori_reward2 + ori_reward3) * ko + vel_reward2 + stability_reward;// + 5*foot.position.y;
        //if (BipedTargetMotion == StyleB.walk) reward += vel_reward1;
        //if (BipedTargetMotion == StyleB.run || BipedTargetMotion == StyleB.jump) reward += vel_reward2;
        AddReward(reward);
        if (Mathf.Abs(EulerTrans(body.eulerAngles[0])) > 20f || Mathf.Abs(EulerTrans(body.eulerAngles[2])) > 20f || tt>=1000)
        {
            if(train)EndEpisode();
            //print(stage);
        }
    }

}
