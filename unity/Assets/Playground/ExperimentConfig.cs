using UnityEngine;
using System.IO;
using System;

/// <summary>
/// P0-3 参数配置化系统
///
/// 功能：
/// 1. 从 JSON 配置文件读取所有实验参数
/// 2. 覆盖 Inspector 中的手动设置（确保一致性）
/// 3. 每次 run 保存生效参数到 results/<run_id>/params.json
///
/// 使用方法：
/// 1. 创建配置文件 Assets/Playground/configs/experiment_config.json
/// 2. 将此脚本挂载到 Robot 对象上（与 RobotRLAgent 同级）
/// 3. 运行时会自动加载并应用配置
/// </summary>
public class ExperimentConfig : MonoBehaviour
{
    [Header("配置文件")]
    [Tooltip("配置文件路径（相对于 Assets/Playground/）")]
    public string configFilePath = "configs/experiment_config.json";

    [Tooltip("是否启用配置系统（false=使用 Inspector 值）")]
    public bool enableConfigSystem = true;

    [Tooltip("每次 run 保存参数快照")]
    public bool saveParamsOnRun = true;

    [Header("运行时信息")]
    public string loadedConfigPath = "";
    public string savedParamsPath = "";

    // 配置数据结构
    [Serializable]
    public class Config
    {
        // L3: 观测配置参数（控制 VectorObservationSize）
        public ObservationParams observation = new ObservationParams();

        // 前馈控制参数
        public FeedforwardParams feedforward = new FeedforwardParams();

        // M1: 甲板 IMU 参数（真前馈传感器）
        public DeckIMUParams deckIMU = new DeckIMUParams();

        // 平台运动参数（旧控制器，向后兼容）
        public PlatformParams platform = new PlatformParams();

        // G1: 波浪谱平台参数（新控制器）
        public WaveSpectrumParams waveSpectrum = new WaveSpectrumParams();

        // 域随机化参数（P2 预留）
        public DomainRandomizationParams domainRandomization = new DomainRandomizationParams();

        // 元信息
        public MetaInfo meta = new MetaInfo();
    }

    /// <summary>
    /// L3: 观测配置参数
    /// 控制 VectorObservationSize 的动态计算
    /// - Baseline (33):     enableCoMObservation=false, enablePredictionObservation=false, enableFeedforwardControl=false
    /// - +CoM (38):         enableCoMObservation=true,  enablePredictionObservation=false, enableFeedforwardControl=false
    /// - +Pred (57):        enableCoMObservation=false, enablePredictionObservation=true,  enableFeedforwardControl=false
    /// - +CoM+Pred+FF (62): enableCoMObservation=true,  enablePredictionObservation=true,  enableFeedforwardControl=true
    /// </summary>
    [Serializable]
    public class ObservationParams
    {
        public bool enableCoMObservation = true;
        public bool enablePredictionObservation = true;
        public bool enableFeedforwardControl = true;  // L3: 前馈控制开关
    }

    [Serializable]
    public class FeedforwardParams
    {
        public float K_roll = 0.1f;
        public float K_pitch = 0.1f;  // P1 新增
        public float max_roll_comp = 8.0f;
        public float max_pitch_comp = 8.0f;  // P1 新增
        public float support_threshold = 0.7f;
        public float filter_alpha = 0.3f;
        public float prediction_horizon = 0.1f;  // 秒
        public int predictor_order = 2;  // 0=zeroth, 1=first, 2=second, 3=oracle
    }

    /// <summary>
    /// M1: 甲板 IMU 配置参数
    /// 控制独立平台运动测量的传感器模式和噪声特性
    /// </summary>
    [Serializable]
    public class DeckIMUParams
    {
        public string sensorMode = "Oracle";  // "Oracle" or "Sensor"
        public float angularVelocityNoise = 0.001f;  // rad/s (~0.06°/s)
        public float accelerationNoise = 0.01f;      // m/s²
        public float attitudeNoise = 0.1f;           // degrees
        public float sensorDelay = 0.01f;            // 秒 (10ms)
    }

    [Serializable]
    public class PlatformParams
    {
        public float rollAmplitude = 0f;
        public float rollFrequency = 0.2f;
        public float rollPhase = 0f;
        public float pitchAmplitude = 0f;
        public float pitchFrequency = 0.2f;
        public float pitchPhase = 0f;
        public float heaveAmplitude = 0f;
        public float heaveFrequency = 0.1f;
        public float heavePhase = 0f;
        public bool enableMotion = true;
        public bool useCurriculumLearning = false;
    }

    /// <summary>
    /// G1: WaveSpectrumPlatformController 参数
    /// </summary>
    [Serializable]
    public class WaveSpectrumParams
    {
        public string waveMode = "JONSWAP";  // "JONSWAP" or "PiersonMoskowitz" or "RegularSine"

        // Irregular wave parameters
        public float significantRollAmplitudeMin = 2f;
        public float significantRollAmplitudeMax = 5f;
        public float significantHeaveAmplitudeMin = 0.1f;
        public float significantHeaveAmplitudeMax = 0.5f;
        public float peakPeriodMin = 6f;
        public float peakPeriodMax = 14f;
        public float gamma = 3.3f;

        // RAO parameters
        public float rollNaturalPeriodMin = 8f;
        public float rollNaturalPeriodMax = 15f;
        public float rollDampingRatioMin = 0.05f;
        public float rollDampingRatioMax = 0.15f;
        public float pitchNaturalPeriodMin = 5f;
        public float pitchNaturalPeriodMax = 10f;
        public float pitchDampingRatioMin = 0.1f;
        public float pitchDampingRatioMax = 0.2f;
        public float heaveNaturalPeriodMin = 4f;
        public float heaveNaturalPeriodMax = 8f;
        public float heaveDampingRatioMin = 0.2f;
        public float heaveDampingRatioMax = 0.4f;

        // Motion scaling
        public float pitchScale = 0.8f;

        // Spectrum discretization
        public int numComponents = 50;
        public float omegaMin = 0.2f;
        public float omegaMax = 3.0f;
        public int waveSeed = -1;

        // Regular wave mode (backward compatibility)
        public float regularRollAmplitude = 5f;
        public float regularRollFrequency = 0.2f;
        public float regularRollPhase = 0f;
        public float regularPitchAmplitude = 5f;
        public float regularPitchFrequency = 0.2f;
        public float regularPitchPhase = 0f;
        public float regularHeaveAmplitude = 0.05f;
        public float regularHeaveFrequency = 0.1f;
        public float regularHeavePhase = 0f;
    }

    [Serializable]
    public class DomainRandomizationParams
    {
        public bool enableMassRandomization = false;
        public float massMultiplierMin = 0.9f;
        public float massMultiplierMax = 1.1f;

        public bool enableCoMRandomization = false;
        public float comOffsetRange = 0.05f;  // 米

        public bool enableFrictionRandomization = false;
        public float frictionMin = 0.4f;
        public float frictionMax = 1.0f;

        public bool enableExternalForce = false;
        public float forceMin = 0f;
        public float forceMax = 50f;
        public float forceIntervalMin = 2f;
        public float forceIntervalMax = 5f;
        public float forceDuration = 0.1f;
    }

    [Serializable]
    public class MetaInfo
    {
        public string experimentName = "";
        public string configVersion = "1.0";
        public string notes = "";
        public string timestamp = "";
    }

    private Config config;
    private string configFullPath;

    void Awake()
    {
        if (!enableConfigSystem)
        {
            Debug.Log("[ExperimentConfig] 配置系统已禁用，使用 Inspector 值");
            return;
        }

        LoadConfig();
        ApplyConfig();

        if (saveParamsOnRun)
        {
            SaveParams();
        }
    }

    /// <summary>
    /// 加载配置文件
    /// </summary>
    void LoadConfig()
    {
        configFullPath = Path.Combine(Application.dataPath, "Playground", configFilePath);

        if (!File.Exists(configFullPath))
        {
            Debug.LogWarning($"[ExperimentConfig] 配置文件不存在: {configFullPath}");
            Debug.LogWarning("[ExperimentConfig] 将创建默认配置文件");
            CreateDefaultConfig();
            return;
        }

        try
        {
            string json = File.ReadAllText(configFullPath);
            config = JsonUtility.FromJson<Config>(json);
            loadedConfigPath = configFullPath;
            Debug.Log($"[ExperimentConfig] ✓ 配置已加载: {configFullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[ExperimentConfig] 配置文件加载失败: {e.Message}");
            config = new Config();
        }
    }

    /// <summary>
    /// 应用配置到各个组件
    /// </summary>
    void ApplyConfig()
    {
        if (config == null)
        {
            Debug.LogWarning("[ExperimentConfig] 配置为空，跳过应用");
            return;
        }

        // 应用前馈控制参数
        var feedforwardController = GetComponent<PredictiveFeedforwardController>();
        if (feedforwardController != null)
        {
            feedforwardController.K_roll = config.feedforward.K_roll;
            feedforwardController.max_roll_comp = config.feedforward.max_roll_comp;
            feedforwardController.support_threshold = config.feedforward.support_threshold;
            feedforwardController.filter_alpha = config.feedforward.filter_alpha;
            Debug.Log($"[ExperimentConfig] ✓ 前馈参数已应用: K_roll={config.feedforward.K_roll}");
        }

        // M1: 应用甲板 IMU 参数
        var deckIMU = GetComponent<DeckIMU>();
        if (deckIMU != null)
        {
            // 解析传感器模式
            if (config.deckIMU.sensorMode == "Oracle")
                deckIMU.mode = DeckIMU.SensorMode.Oracle;
            else
                deckIMU.mode = DeckIMU.SensorMode.Sensor;

            deckIMU.angularVelocityNoise = config.deckIMU.angularVelocityNoise;
            deckIMU.accelerationNoise = config.deckIMU.accelerationNoise;
            deckIMU.attitudeNoise = config.deckIMU.attitudeNoise;
            deckIMU.sensorDelay = config.deckIMU.sensorDelay;

            Debug.Log($"[ExperimentConfig] ✓ 甲板 IMU 参数已应用: mode={config.deckIMU.sensorMode}, " +
                     $"delay={config.deckIMU.sensorDelay * 1000:F1}ms");
        }

        // G1: 应用平台参数 - 支持新旧两种控制器
        // 优先尝试在父级 area 中查找（Training Area 架构）
        Transform areaRoot = transform.parent;
        GameObject ground = null;

        if (areaRoot != null)
        {
            // 在同一 area 中查找 Ground
            var groundTransform = areaRoot.Find("Ground");
            if (groundTransform != null)
            {
                ground = groundTransform.gameObject;
            }
        }

        if (ground == null)
        {
            // Fallback: 全局查找（单 agent 场景）
            ground = GameObject.Find("Ground");
        }

        if (ground != null)
        {
            // 尝试新控制器
            var waveController = ground.GetComponent<WaveSpectrumPlatformController>();
            if (waveController != null)
            {
                // 解析 waveMode
                if (config.waveSpectrum.waveMode == "JONSWAP")
                    waveController.waveMode = WaveMode.JONSWAP;
                else if (config.waveSpectrum.waveMode == "PiersonMoskowitz" || config.waveSpectrum.waveMode == "PM")
                    waveController.waveMode = WaveMode.PiersonMoskowitz;
                else
                    waveController.waveMode = WaveMode.RegularSine;

                // Irregular wave parameters
                waveController.significantRollAmplitudeRange = new Vector2(
                    config.waveSpectrum.significantRollAmplitudeMin,
                    config.waveSpectrum.significantRollAmplitudeMax);
                waveController.significantHeaveAmplitudeRange = new Vector2(
                    config.waveSpectrum.significantHeaveAmplitudeMin,
                    config.waveSpectrum.significantHeaveAmplitudeMax);
                waveController.peakPeriodRange = new Vector2(
                    config.waveSpectrum.peakPeriodMin,
                    config.waveSpectrum.peakPeriodMax);
                waveController.gamma = config.waveSpectrum.gamma;

                // RAO parameters
                waveController.rollNaturalPeriodRange = new Vector2(
                    config.waveSpectrum.rollNaturalPeriodMin,
                    config.waveSpectrum.rollNaturalPeriodMax);
                waveController.rollDampingRatioRange = new Vector2(
                    config.waveSpectrum.rollDampingRatioMin,
                    config.waveSpectrum.rollDampingRatioMax);
                waveController.pitchNaturalPeriodRange = new Vector2(
                    config.waveSpectrum.pitchNaturalPeriodMin,
                    config.waveSpectrum.pitchNaturalPeriodMax);
                waveController.pitchDampingRatioRange = new Vector2(
                    config.waveSpectrum.pitchDampingRatioMin,
                    config.waveSpectrum.pitchDampingRatioMax);
                waveController.heaveNaturalPeriodRange = new Vector2(
                    config.waveSpectrum.heaveNaturalPeriodMin,
                    config.waveSpectrum.heaveNaturalPeriodMax);
                waveController.heaveDampingRatioRange = new Vector2(
                    config.waveSpectrum.heaveDampingRatioMin,
                    config.waveSpectrum.heaveDampingRatioMax);

                // Motion scaling
                waveController.pitchScale = config.waveSpectrum.pitchScale;

                // Spectrum discretization
                waveController.numComponents = config.waveSpectrum.numComponents;
                waveController.omegaMin = config.waveSpectrum.omegaMin;
                waveController.omegaMax = config.waveSpectrum.omegaMax;
                waveController.waveSeed = config.waveSpectrum.waveSeed;

                // Regular wave mode
                waveController.regularRollAmplitude = config.waveSpectrum.regularRollAmplitude;
                waveController.regularRollFrequency = config.waveSpectrum.regularRollFrequency;
                waveController.regularRollPhase = config.waveSpectrum.regularRollPhase;
                waveController.regularPitchAmplitude = config.waveSpectrum.regularPitchAmplitude;
                waveController.regularPitchFrequency = config.waveSpectrum.regularPitchFrequency;
                waveController.regularPitchPhase = config.waveSpectrum.regularPitchPhase;
                waveController.regularHeaveAmplitude = config.waveSpectrum.regularHeaveAmplitude;
                waveController.regularHeaveFrequency = config.waveSpectrum.regularHeaveFrequency;
                waveController.regularHeavePhase = config.waveSpectrum.regularHeavePhase;

                Debug.Log($"[ExperimentConfig] ✓ WaveSpectrum 参数已应用: " +
                          $"mode={config.waveSpectrum.waveMode}, " +
                          $"Hs={config.waveSpectrum.significantRollAmplitudeMin}-{config.waveSpectrum.significantRollAmplitudeMax}°, " +
                          $"Tp={config.waveSpectrum.peakPeriodMin}-{config.waveSpectrum.peakPeriodMax}s");
            }
            else
            {
                // Fallback: 旧控制器
                var platformController = ground.GetComponent<PlatformMotionController>();
                if (platformController != null)
                {
                    platformController.rollAmplitude = config.platform.rollAmplitude;
                    platformController.rollFrequency = config.platform.rollFrequency;
                    platformController.rollPhase = config.platform.rollPhase;
                    platformController.pitchAmplitude = config.platform.pitchAmplitude;
                    platformController.pitchFrequency = config.platform.pitchFrequency;
                    platformController.pitchPhase = config.platform.pitchPhase;
                    platformController.heaveAmplitude = config.platform.heaveAmplitude;
                    platformController.heaveFrequency = config.platform.heaveFrequency;
                    platformController.heavePhase = config.platform.heavePhase;
                    platformController.enableMotion = config.platform.enableMotion;
                    platformController.useCurriculumLearning = config.platform.useCurriculumLearning;

                    Debug.Log($"[ExperimentConfig] ✓ 平台参数已应用 (旧控制器): " +
                              $"roll={config.platform.rollAmplitude}°@{config.platform.rollFrequency}Hz, " +
                              $"pitch={config.platform.pitchAmplitude}°@{config.platform.pitchFrequency}Hz");
                }
            }
        }
    }

    /// <summary>
    /// 保存当前生效参数到 results/<run_id>/params.json
    /// </summary>
    void SaveParams()
    {
        if (config == null)
        {
            config = CollectCurrentParams();
        }

        // 添加时间戳
        config.meta.timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

        // 生成保存路径
        string runId = config.meta.experimentName;
        if (string.IsNullOrEmpty(runId))
        {
            runId = $"run_{DateTime.Now:yyyyMMdd_HHmmss}";
        }

        string resultsDir = Path.Combine(Application.dataPath, "Playground", "results", runId);
        Directory.CreateDirectory(resultsDir);

        string paramsPath = Path.Combine(resultsDir, "params.json");

        try
        {
            string json = JsonUtility.ToJson(config, true);
            File.WriteAllText(paramsPath, json);
            savedParamsPath = paramsPath;
            Debug.Log($"[ExperimentConfig] ✓ 参数已保存: {paramsPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[ExperimentConfig] 参数保存失败: {e.Message}");
        }
    }

    /// <summary>
    /// G1: 收集当前各组件的参数（用于未加载配置文件时）
    /// </summary>
    Config CollectCurrentParams()
    {
        var collected = new Config();

        // L3: 收集观测配置
        var agent = GetComponent<RobotRLAgent>();
        if (agent != null)
        {
            // 使用反射读取观测开关
            var agentType = agent.GetType();
            var enableCoMField = agentType.GetField("enableCoMObservation");
            var enablePredField = agentType.GetField("enablePredictionObservation");
            var enableFFField = agentType.GetField("enableFeedforwardControl");

            if (enableCoMField != null)
            {
                collected.observation.enableCoMObservation = (bool)enableCoMField.GetValue(agent);
            }
            if (enablePredField != null)
            {
                collected.observation.enablePredictionObservation = (bool)enablePredField.GetValue(agent);
            }
            if (enableFFField != null)
            {
                collected.observation.enableFeedforwardControl = (bool)enableFFField.GetValue(agent);
            }
        }

        // M1: 收集甲板 IMU 配置
        var deckIMU = GetComponent<DeckIMU>();
        if (deckIMU != null)
        {
            collected.deckIMU.sensorMode = deckIMU.mode.ToString();
            collected.deckIMU.angularVelocityNoise = deckIMU.angularVelocityNoise;
            collected.deckIMU.accelerationNoise = deckIMU.accelerationNoise;
            collected.deckIMU.attitudeNoise = deckIMU.attitudeNoise;
            collected.deckIMU.sensorDelay = deckIMU.sensorDelay;
        }

        var feedforwardController = GetComponent<PredictiveFeedforwardController>();
        if (feedforwardController != null)
        {
            collected.feedforward.K_roll = feedforwardController.K_roll;
            collected.feedforward.max_roll_comp = feedforwardController.max_roll_comp;
            collected.feedforward.support_threshold = feedforwardController.support_threshold;
            collected.feedforward.filter_alpha = feedforwardController.filter_alpha;
        }

        // 查找平台（优先在同一 area，Fallback 全局）
        Transform areaRoot = transform.parent;
        GameObject ground = null;

        if (areaRoot != null)
        {
            var groundTransform = areaRoot.Find("Ground");
            if (groundTransform != null)
            {
                ground = groundTransform.gameObject;
            }
        }

        if (ground == null)
        {
            ground = GameObject.Find("Ground");
        }

        if (ground != null)
        {
            // 尝试新控制器
            var waveController = ground.GetComponent<WaveSpectrumPlatformController>();
            if (waveController != null)
            {
                // 收集 WaveSpectrum 参数
                collected.waveSpectrum.waveMode = waveController.waveMode.ToString();
                collected.waveSpectrum.significantRollAmplitudeMin = waveController.significantRollAmplitudeRange.x;
                collected.waveSpectrum.significantRollAmplitudeMax = waveController.significantRollAmplitudeRange.y;
                collected.waveSpectrum.significantHeaveAmplitudeMin = waveController.significantHeaveAmplitudeRange.x;
                collected.waveSpectrum.significantHeaveAmplitudeMax = waveController.significantHeaveAmplitudeRange.y;
                collected.waveSpectrum.peakPeriodMin = waveController.peakPeriodRange.x;
                collected.waveSpectrum.peakPeriodMax = waveController.peakPeriodRange.y;
                collected.waveSpectrum.gamma = waveController.gamma;

                collected.waveSpectrum.rollNaturalPeriodMin = waveController.rollNaturalPeriodRange.x;
                collected.waveSpectrum.rollNaturalPeriodMax = waveController.rollNaturalPeriodRange.y;
                collected.waveSpectrum.rollDampingRatioMin = waveController.rollDampingRatioRange.x;
                collected.waveSpectrum.rollDampingRatioMax = waveController.rollDampingRatioRange.y;
                collected.waveSpectrum.pitchNaturalPeriodMin = waveController.pitchNaturalPeriodRange.x;
                collected.waveSpectrum.pitchNaturalPeriodMax = waveController.pitchNaturalPeriodRange.y;
                collected.waveSpectrum.pitchDampingRatioMin = waveController.pitchDampingRatioRange.x;
                collected.waveSpectrum.pitchDampingRatioMax = waveController.pitchDampingRatioRange.y;
                collected.waveSpectrum.heaveNaturalPeriodMin = waveController.heaveNaturalPeriodRange.x;
                collected.waveSpectrum.heaveNaturalPeriodMax = waveController.heaveNaturalPeriodRange.y;
                collected.waveSpectrum.heaveDampingRatioMin = waveController.heaveDampingRatioRange.x;
                collected.waveSpectrum.heaveDampingRatioMax = waveController.heaveDampingRatioRange.y;

                collected.waveSpectrum.pitchScale = waveController.pitchScale;
                collected.waveSpectrum.numComponents = waveController.numComponents;
                collected.waveSpectrum.omegaMin = waveController.omegaMin;
                collected.waveSpectrum.omegaMax = waveController.omegaMax;
                collected.waveSpectrum.waveSeed = waveController.waveSeed;

                collected.waveSpectrum.regularRollAmplitude = waveController.regularRollAmplitude;
                collected.waveSpectrum.regularRollFrequency = waveController.regularRollFrequency;
                collected.waveSpectrum.regularRollPhase = waveController.regularRollPhase;
                collected.waveSpectrum.regularPitchAmplitude = waveController.regularPitchAmplitude;
                collected.waveSpectrum.regularPitchFrequency = waveController.regularPitchFrequency;
                collected.waveSpectrum.regularPitchPhase = waveController.regularPitchPhase;
                collected.waveSpectrum.regularHeaveAmplitude = waveController.regularHeaveAmplitude;
                collected.waveSpectrum.regularHeaveFrequency = waveController.regularHeaveFrequency;
                collected.waveSpectrum.regularHeavePhase = waveController.regularHeavePhase;
            }
            else
            {
                // Fallback: 旧控制器
                var platformController = ground.GetComponent<PlatformMotionController>();
                if (platformController != null)
                {
                    collected.platform.rollAmplitude = platformController.rollAmplitude;
                    collected.platform.rollFrequency = platformController.rollFrequency;
                    collected.platform.rollPhase = platformController.rollPhase;
                    collected.platform.pitchAmplitude = platformController.pitchAmplitude;
                    collected.platform.pitchFrequency = platformController.pitchFrequency;
                    collected.platform.pitchPhase = platformController.pitchPhase;
                    collected.platform.heaveAmplitude = platformController.heaveAmplitude;
                    collected.platform.heaveFrequency = platformController.heaveFrequency;
                    collected.platform.heavePhase = platformController.heavePhase;
                    collected.platform.enableMotion = platformController.enableMotion;
                    collected.platform.useCurriculumLearning = platformController.useCurriculumLearning;
                }
            }
        }

        return collected;
    }

    /// <summary>
    /// 创建默认配置文件
    /// </summary>
    void CreateDefaultConfig()
    {
        config = new Config();
        config.meta.experimentName = "default";
        config.meta.notes = "Auto-generated default configuration";

        string configDir = Path.GetDirectoryName(configFullPath);
        Directory.CreateDirectory(configDir);

        try
        {
            string json = JsonUtility.ToJson(config, true);
            File.WriteAllText(configFullPath, json);
            Debug.Log($"[ExperimentConfig] ✓ 默认配置已创建: {configFullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[ExperimentConfig] 创建默认配置失败: {e.Message}");
        }
    }

    /// <summary>
    /// 公开接口：获取当前配置
    /// </summary>
    public Config GetConfig()
    {
        return config;
    }
}
