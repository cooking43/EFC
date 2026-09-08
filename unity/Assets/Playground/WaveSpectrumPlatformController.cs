using UnityEngine;
using System.Collections.Generic;
using Unity.MLAgents;  // AC2: 用于课程学习参数读取

/// <summary>
/// P1: 不规则波浪谱平台控制器
///
/// 实现 JONSWAP/PM 波浪谱 + 通用二阶 RAO + 随机相位叠加
///
/// 关键特性:
/// - 不绑定具体船型（通用 RAO 参数可配置）
/// - 直接控制 significantRollAmplitude 和 peakPeriod
/// - 每 episode 重采样相位和海况参数
/// - roll/pitch/heave 独立相位集（耦合运动）
/// - 保留 RegularSine 模式用于对照
/// - 暴露解析真值供 oracle 预测器使用
///
/// G1 坐标系约定（实测验证）:
/// Unity 左手系，+roll（绕 Z 轴正向旋转）→ 右侧（+X）上升，左侧（-X）下沉
/// </summary>
public class WaveSpectrumPlatformController : MonoBehaviour
{
    [Header("=== Wave Mode ===")]
    public WaveMode waveMode = WaveMode.JONSWAP;

    [Header("=== Irregular Wave Parameters ===")]
    [Tooltip("显著横摇幅度范围 (度)，例如 [1, 5] 训练 / [1, 12] OOD")]
    public Vector2 significantRollAmplitudeRange = new Vector2(2f, 5f);

    [Tooltip("谱峰周期范围 (秒)，例如 [6, 14]")]
    public Vector2 peakPeriodRange = new Vector2(6f, 14f);

    [Tooltip("JONSWAP 谱增强因子。3.3 = JONSWAP, 1.0 = PM 谱")]
    public float gamma = 3.3f;

    [Header("=== RAO Parameters (Vessel-Agnostic) ===")]
    [Tooltip("横摇固有周期范围 (秒)，典型: [8, 15]")]
    public Vector2 rollNaturalPeriodRange = new Vector2(8f, 15f);

    [Tooltip("横摇阻尼比范围，典型: [0.05, 0.15]")]
    public Vector2 rollDampingRatioRange = new Vector2(0.05f, 0.15f);

    [Tooltip("纵摇固有周期范围 (秒)，典型: [5, 10]")]
    public Vector2 pitchNaturalPeriodRange = new Vector2(5f, 10f);

    [Tooltip("纵摇阻尼比范围，典型: [0.1, 0.2]")]
    public Vector2 pitchDampingRatioRange = new Vector2(0.1f, 0.2f);

    [Tooltip("升沉固有周期范围 (秒)，典型: [4, 8]")]
    public Vector2 heaveNaturalPeriodRange = new Vector2(4f, 8f);

    [Tooltip("升沉阻尼比范围，典型: [0.2, 0.4]")]
    public Vector2 heaveDampingRatioRange = new Vector2(0.2f, 0.4f);

    [Header("=== Motion Scaling ===")]
    [Tooltip("纵摇相对横摇的强度比例")]
    public float pitchScale = 0.8f;

    [Tooltip("显著升沉幅度范围 (米)，例如 [0.1, 0.5]")]
    public Vector2 significantHeaveAmplitudeRange = new Vector2(0.1f, 0.5f);

    [Header("=== Spectrum Discretization ===")]
    [Tooltip("频率分量数量")]
    public int numComponents = 50;

    [Tooltip("最小频率 (rad/s)")]
    public float omegaMin = 0.2f;

    [Tooltip("最大频率 (rad/s)")]
    public float omegaMax = 3.0f;

    [Tooltip("波浪随机种子，-1 = 每 episode 重采样")]
    public int waveSeed = -1;

    [Header("=== Regular Wave Mode (Backward Compatibility) ===")]
    public float regularRollAmplitude = 5f;  // deg
    public float regularRollFrequency = 0.2f; // Hz
    public float regularRollPhase = 0f;
    public float regularPitchAmplitude = 5f;
    public float regularPitchFrequency = 0.2f;
    public float regularPitchPhase = 0f;
    public float regularHeaveAmplitude = 0.05f; // m
    public float regularHeaveFrequency = 0.1f;
    public float regularHeavePhase = 0f;

    [Header("=== Runtime Status ===")]
    [SerializeField, Tooltip("当前显著横摇幅度 (度)")]
    private float currentSignificantRollAmplitude;

    [SerializeField, Tooltip("当前显著升沉幅度 (米)")]
    private float currentSignificantHeaveAmplitude;

    [SerializeField, Tooltip("当前谱峰周期 (秒)")]
    private float currentPeakPeriod;

    [SerializeField, Tooltip("当前波浪种子")]
    private int currentSeed;

    // H3: Public getters for test verification
    public float CurrentSignificantRollAmplitude => currentSignificantRollAmplitude;
    public float CurrentSignificantHeaveAmplitude => currentSignificantHeaveAmplitude;
    public float CurrentPeakPeriod => currentPeakPeriod;
    public int CurrentSeed => currentSeed;

    // 内部状态
    private Transform platformTransform;
    private Rigidbody platformRigidbody;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private float time = 0f;

    // H2: Episode 开始时间（用于幅度平滑爬升）
    private float episodeStartTime = 0f;
    private const float AMPLITUDE_RAMP_DURATION = 0.5f;  // 0.5s 爬升时间

    // 不规则波参数（当前 episode）
    private float[] omegas;              // 频率分量 (rad/s)
    private float[] rollPhases;          // 横摇相位
    private float[] pitchPhases;         // 纵摇相位
    private float[] heavePhases;         // 升沉相位
    private float[] rollAmplitudes;      // 横摇振幅
    private float[] pitchAmplitudes;     // 纵摇振幅
    private float[] heaveAmplitudes;     // 升沉振幅

    // RAO 参数（当前 episode）
    private float rollOmegaN;            // 横摇固有频率 (rad/s)
    private float rollZeta;              // 横摇阻尼比
    private float pitchOmegaN;
    private float pitchZeta;
    private float heaveOmegaN;
    private float heaveZeta;

    // 缓存的导数（供 oracle 使用）
    private Vector3 cachedAngularVelocity;    // deg/s
    private Vector3 cachedAngularAccel;       // deg/s²
    private Vector3 cachedLinearAccel;        // m/s²

    // H1 修复：改用 Awake() 而非 Start()，确保在 OnEpisodeBegin() 调用 ResetMotion() 前完成初始化
    void Awake()
    {
        platformTransform = transform;
        initialPosition = platformTransform.position;
        initialRotation = platformTransform.rotation;

        // B2/F3 修复：获取或添加 Rigidbody（kinematic）
        platformRigidbody = GetComponent<Rigidbody>();
        if (platformRigidbody == null)
        {
            platformRigidbody = gameObject.AddComponent<Rigidbody>();
            platformRigidbody.isKinematic = true;
            Debug.LogWarning("[WaveSpectrumPlatformController] 平台缺少 Rigidbody，已自动添加 kinematic Rigidbody");
        }
        else if (!platformRigidbody.isKinematic)
        {
            // F3 修复：强制设为 kinematic，否则 MoveRotation 会和物理引擎打架
            Debug.LogWarning("[WaveSpectrumPlatformController] 平台 Rigidbody 非 kinematic，已强制修正");
            platformRigidbody.isKinematic = true;
        }
    }

    void Start()
    {
        // 初始化波浪参数
        if (waveMode != WaveMode.RegularSine)
        {
            ResampleWave(waveSeed);
        }
    }

    void FixedUpdate()
    {
        time += Time.fixedDeltaTime;

        Vector3 euler;  // deg
        Vector3 position;

        if (waveMode == WaveMode.RegularSine)
        {
            // 规则波模式（向后兼容）
            float roll = regularRollAmplitude * Mathf.Sin(2f * Mathf.PI * regularRollFrequency * time + regularRollPhase * Mathf.Deg2Rad);
            float pitch = regularPitchAmplitude * Mathf.Sin(2f * Mathf.PI * regularPitchFrequency * time + regularPitchPhase * Mathf.Deg2Rad);
            float heave = regularHeaveAmplitude * Mathf.Sin(2f * Mathf.PI * regularHeaveFrequency * time + regularHeavePhase * Mathf.Deg2Rad);

            euler = new Vector3(pitch, 0f, roll);
            position = initialPosition + new Vector3(0f, heave, 0f);

            // 简单数值导数（规则波）
            float rollOmega = 2f * Mathf.PI * regularRollFrequency;
            float pitchOmega = 2f * Mathf.PI * regularPitchFrequency;
            float heaveOmega = 2f * Mathf.PI * regularHeaveFrequency;

            cachedAngularVelocity = new Vector3(
                regularPitchAmplitude * pitchOmega * Mathf.Cos(pitchOmega * time + regularPitchPhase * Mathf.Deg2Rad),
                0f,
                regularRollAmplitude * rollOmega * Mathf.Cos(rollOmega * time + regularRollPhase * Mathf.Deg2Rad)
            );

            cachedAngularAccel = new Vector3(
                -regularPitchAmplitude * pitchOmega * pitchOmega * Mathf.Sin(pitchOmega * time + regularPitchPhase * Mathf.Deg2Rad),
                0f,
                -regularRollAmplitude * rollOmega * rollOmega * Mathf.Sin(rollOmega * time + regularRollPhase * Mathf.Deg2Rad)
            );

            cachedLinearAccel = new Vector3(
                0f,
                -regularHeaveAmplitude * heaveOmega * heaveOmega * Mathf.Sin(heaveOmega * time + regularHeavePhase * Mathf.Deg2Rad),
                0f
            );
        }
        else
        {
            // H2/J3: 计算幅度平滑爬升系数及其导数（episode 开头 0.5s 从 0 爬升到 1）
            float timeSinceEpisodeStart = time - episodeStartTime;
            float rampFactor = 1f;
            float rampFactorDerivative = 0f;  // J3: dr/dt
            float rampFactorSecondDerivative = 0f;  // J3: d²r/dt²

            if (timeSinceEpisodeStart < AMPLITUDE_RAMP_DURATION)
            {
                // Smoothstep: r(t) = 3t² - 2t³
                float t01 = Mathf.Clamp01(timeSinceEpisodeStart / AMPLITUDE_RAMP_DURATION);
                rampFactor = t01 * t01 * (3f - 2f * t01);

                // J3: Smoothstep 导数
                // r'(t) = (6t - 6t²) / T_ramp
                // r''(t) = (6 - 12t) / T_ramp²
                rampFactorDerivative = (6f * t01 - 6f * t01 * t01) / AMPLITUDE_RAMP_DURATION;
                rampFactorSecondDerivative = (6f - 12f * t01) / (AMPLITUDE_RAMP_DURATION * AMPLITUDE_RAMP_DURATION);
            }

            // 不规则波模式（随机相位叠加）
            // J3: 分别计算 s(t), ṡ(t), s̈(t)，然后应用乘积法则
            float roll = 0f, pitch = 0f, heave = 0f;
            float rollBase = 0f, pitchBase = 0f, heaveBase = 0f;  // J3: s(t) without ramp
            float rollVelBase = 0f, pitchVelBase = 0f, heaveVelBase = 0f;  // J3: ṡ(t) without ramp
            float rollAccBase = 0f, pitchAccBase = 0f, heaveAccBase = 0f;  // J3: s̈(t) without ramp

            for (int i = 0; i < numComponents; i++)
            {
                float omega = omegas[i];
                float t = time;

                // J3: 先计算基础波浪分量（无 ramp）
                // 横摇
                float rollPhase = omega * t + rollPhases[i];
                float rollComponent = rollAmplitudes[i] * Mathf.Cos(rollPhase);
                float rollVelComponent = -rollAmplitudes[i] * omega * Mathf.Sin(rollPhase);
                float rollAccComponent = -rollAmplitudes[i] * omega * omega * Mathf.Cos(rollPhase);

                rollBase += rollComponent;
                rollVelBase += rollVelComponent;
                rollAccBase += rollAccComponent;

                // 纵摇
                float pitchPhase = omega * t + pitchPhases[i];
                float pitchComponent = pitchAmplitudes[i] * Mathf.Cos(pitchPhase);
                float pitchVelComponent = -pitchAmplitudes[i] * omega * Mathf.Sin(pitchPhase);
                float pitchAccComponent = -pitchAmplitudes[i] * omega * omega * Mathf.Cos(pitchPhase);

                pitchBase += pitchComponent;
                pitchVelBase += pitchVelComponent;
                pitchAccBase += pitchAccComponent;

                // 升沉
                float heavePhase = omega * t + heavePhases[i];
                float heaveComponent = heaveAmplitudes[i] * Mathf.Cos(heavePhase);
                float heaveVelComponent = -heaveAmplitudes[i] * omega * Mathf.Sin(heavePhase);
                float heaveAccComponent = -heaveAmplitudes[i] * omega * omega * Mathf.Cos(heavePhase);

                heaveBase += heaveComponent;
                heaveVelBase += heaveVelComponent;
                heaveAccBase += heaveAccComponent;
            }

            // J3: 应用乘积法则
            // θ(t) = r(t) · s(t)
            roll = rampFactor * rollBase;
            pitch = rampFactor * pitchBase;
            heave = rampFactor * heaveBase;

            // θ̇(t) = r'(t) · s(t) + r(t) · ṡ(t)
            float rollVel = rampFactorDerivative * rollBase + rampFactor * rollVelBase;
            float pitchVel = rampFactorDerivative * pitchBase + rampFactor * pitchVelBase;
            float heaveVel = rampFactorDerivative * heaveBase + rampFactor * heaveVelBase;

            // θ̈(t) = r''(t) · s(t) + 2 · r'(t) · ṡ(t) + r(t) · s̈(t)
            float rollAcc = rampFactorSecondDerivative * rollBase + 2f * rampFactorDerivative * rollVelBase + rampFactor * rollAccBase;
            float pitchAcc = rampFactorSecondDerivative * pitchBase + 2f * rampFactorDerivative * pitchVelBase + rampFactor * pitchAccBase;
            float heaveAcc = rampFactorSecondDerivative * heaveBase + 2f * rampFactorDerivative * heaveVelBase + rampFactor * heaveAccBase;

            euler = new Vector3(pitch, 0f, roll);
            position = initialPosition + new Vector3(0f, heave, 0f);

            // P1a 修复：rollVel/pitchVel 已是 deg/s（度幅值 × rad/s），不需要再乘 Rad2Deg
            // rollAmplitudes[i]（度）× omegas[i]（rad/s）= deg/s
            cachedAngularVelocity = new Vector3(pitchVel, 0f, rollVel);
            cachedAngularAccel = new Vector3(pitchAcc, 0f, rollAcc);
            cachedLinearAccel = new Vector3(0f, heaveAcc, 0f);
        }

        // B2 修复：使用 Rigidbody.MoveRotation/MovePosition 而非直接设置 Transform
        // 这确保 kinematic body 正确更新接触速度，使机器人被平台带动而非滑动
        platformRigidbody.MoveRotation(initialRotation * Quaternion.Euler(euler));
        platformRigidbody.MovePosition(position);
    }

    /// <summary>
    /// P1: 重采样波浪参数（每 episode 调用）
    /// </summary>
    public void ResampleWave(int seed = -1)
    {
        // 使用提供的种子或生成新种子
        if (seed < 0)
        {
            currentSeed = Random.Range(0, 1000000);
        }
        else
        {
            currentSeed = seed;
        }

        Random.State oldState = Random.state;
        Random.InitState(currentSeed);

        // AC2: 课程学习支持 - 检查 environment_parameters 是否指定了固定的 significant_roll_amplitude
        // 如果指定了（课程期间），使用该固定值；否则从 significantRollAmplitudeRange 随机采样
        float curriculumAmplitude = Academy.Instance.EnvironmentParameters.GetWithDefault("significant_roll_amplitude", -1f);
        if (curriculumAmplitude >= 0f)
        {
            // 课程模式：使用环境参数指定的固定值
            currentSignificantRollAmplitude = curriculumAmplitude;
        }
        else
        {
            // 正常模式：从范围随机采样
            currentSignificantRollAmplitude = Random.Range(significantRollAmplitudeRange.x, significantRollAmplitudeRange.y);
        }
        currentSignificantHeaveAmplitude = Random.Range(significantHeaveAmplitudeRange.x, significantHeaveAmplitudeRange.y);
        currentPeakPeriod = Random.Range(peakPeriodRange.x, peakPeriodRange.y);
        float omegaP = 2f * Mathf.PI / currentPeakPeriod;

        // 随机化 RAO 参数
        float rollTn = Random.Range(rollNaturalPeriodRange.x, rollNaturalPeriodRange.y);
        rollOmegaN = 2f * Mathf.PI / rollTn;
        rollZeta = Random.Range(rollDampingRatioRange.x, rollDampingRatioRange.y);

        float pitchTn = Random.Range(pitchNaturalPeriodRange.x, pitchNaturalPeriodRange.y);
        pitchOmegaN = 2f * Mathf.PI / pitchTn;
        pitchZeta = Random.Range(pitchDampingRatioRange.x, pitchDampingRatioRange.y);

        float heaveTn = Random.Range(heaveNaturalPeriodRange.x, heaveNaturalPeriodRange.y);
        heaveOmegaN = 2f * Mathf.PI / heaveTn;
        heaveZeta = Random.Range(heaveDampingRatioRange.x, heaveDampingRatioRange.y);

        // 离散化频率
        omegas = new float[numComponents];
        float deltaOmega = (omegaMax - omegaMin) / numComponents;

        for (int i = 0; i < numComponents; i++)
        {
            omegas[i] = omegaMin + (i + 0.5f) * deltaOmega;
        }

        // 计算 JONSWAP/PM 谱 + RAO 响应谱
        // 注意：Hs 这里用度数值是无害的，仅作为 Hs² 的整体因子，
        // 被后面的幅度标定吸收（L324）。最终输出的 roll 是度数。
        float Hs = currentSignificantRollAmplitude / 2f;  // 临时值，后面会标定

        float[] S_omega = new float[numComponents];
        float[] S_roll = new float[numComponents];
        float[] S_pitch = new float[numComponents];
        float[] S_heave = new float[numComponents];

        for (int i = 0; i < numComponents; i++)
        {
            float omega = omegas[i];

            // JONSWAP 波浪谱
            float sigma = (omega <= omegaP) ? 0.07f : 0.09f;
            float r = Mathf.Exp(-Mathf.Pow(omega - omegaP, 2f) / (2f * sigma * sigma * omegaP * omegaP));
            float gammaFactor = Mathf.Pow(gamma, r) * (1f - 0.287f * Mathf.Log(gamma));

            S_omega[i] = (5f / 16f) * Hs * Hs * Mathf.Pow(omegaP, 4f) * Mathf.Pow(omega, -5f)
                         * Mathf.Exp(-1.25f * Mathf.Pow(omegaP / omega, 4f))
                         * gammaFactor;

            // RAO 传递函数（通用二阶）
            float H_roll = ComputeRAO(omega, rollOmegaN, rollZeta);
            float H_pitch = ComputeRAO(omega, pitchOmegaN, pitchZeta);
            float H_heave = ComputeRAO(omega, heaveOmegaN, heaveZeta);

            // 运动响应谱
            S_roll[i] = H_roll * H_roll * S_omega[i];
            S_pitch[i] = H_pitch * H_pitch * S_omega[i];
            S_heave[i] = H_heave * H_heave * S_omega[i];
        }

        // 零阶矩（标准差）
        float m0_roll = 0f, m0_pitch = 0f, m0_heave = 0f;
        for (int i = 0; i < numComponents; i++)
        {
            m0_roll += S_roll[i] * deltaOmega;
            m0_pitch += S_pitch[i] * deltaOmega;
            m0_heave += S_heave[i] * deltaOmega;
        }

        float sigma_roll = Mathf.Sqrt(m0_roll);
        float sigma_pitch = Mathf.Sqrt(m0_pitch);
        float sigma_heave = Mathf.Sqrt(m0_heave);

        // B1 修复：幅度标定（roll/pitch 单位是度，不需要 Deg2Rad）
        // 合成的 roll 值直接送到 Quaternion.Euler（期望度数）
        // B3 修复：heave 独立标定（单位是米，量纲与 roll 不同）
        float scaleRoll = (sigma_roll > 1e-6f) ? (currentSignificantRollAmplitude / (2f * sigma_roll)) : 1f;
        float scalePitch = scaleRoll * pitchScale;
        float scaleHeave = (sigma_heave > 1e-6f) ? (currentSignificantHeaveAmplitude / (2f * sigma_heave)) : 1f;

        // 随机相位叠加振幅
        rollAmplitudes = new float[numComponents];
        pitchAmplitudes = new float[numComponents];
        heaveAmplitudes = new float[numComponents];
        rollPhases = new float[numComponents];
        pitchPhases = new float[numComponents];
        heavePhases = new float[numComponents];

        for (int i = 0; i < numComponents; i++)
        {
            // 振幅 = sqrt(2 * S * Δω)
            rollAmplitudes[i] = Mathf.Sqrt(2f * S_roll[i] * deltaOmega) * scaleRoll;
            pitchAmplitudes[i] = Mathf.Sqrt(2f * S_pitch[i] * deltaOmega) * scalePitch;
            heaveAmplitudes[i] = Mathf.Sqrt(2f * S_heave[i] * deltaOmega) * scaleHeave;

            // 独立随机相位
            rollPhases[i] = Random.Range(0f, 2f * Mathf.PI);
            pitchPhases[i] = Random.Range(0f, 2f * Mathf.PI);
            heavePhases[i] = Random.Range(0f, 2f * Mathf.PI);
        }

        Random.state = oldState;

        // B4 修复：不重置 time（多 clone 共享平台会造成竞态）
        // 换相位集后时序已经全新，重置反而制造不连续并给 ArticulationBody 冲击
        // time 保持连续递增

        Debug.Log($"[WaveSpectrumPlatformController] 波浪重采样完成");
        Debug.Log($"  种子: {currentSeed}");
        Debug.Log($"  显著横摇幅度: {currentSignificantRollAmplitude:F2}°");
        Debug.Log($"  谱峰周期: {currentPeakPeriod:F2} s");
        Debug.Log($"  横摇 Tn={rollTn:F2}s ζ={rollZeta:F3}");
        Debug.Log($"  纵摇 Tn={pitchTn:F2}s ζ={pitchZeta:F3}");
        Debug.Log($"  升沉 Tn={heaveTn:F2}s ζ={heaveZeta:F3}");
    }

    /// <summary>
    /// 计算通用二阶 RAO 传递函数幅值
    /// </summary>
    private float ComputeRAO(float omega, float omegaN, float zeta)
    {
        float num = omegaN * omegaN;
        float denom1 = omegaN * omegaN - omega * omega;
        float denom2 = 2f * zeta * omegaN * omega;
        float denom = Mathf.Sqrt(denom1 * denom1 + denom2 * denom2);

        return (denom > 1e-6f) ? (num / denom) : 0f;
    }

    /// <summary>
    /// H1 修复：重置平台到初始状态（边界传送）
    ///
    /// Training Area 架构：
    /// - 每个 agent 拥有独立平台，可以安全重置
    /// - 配合 ResampleWave() 在 episode 边界随机化海况
    ///
    /// H1 关键修正：边界传送用直接赋值，不用 MoveRotation/MovePosition
    /// - MoveRotation 会让物理引擎从位移差反算角速度 → episode 边界姿态跳变被算成巨大尖峰
    /// - 直接赋值 rb.rotation/position 不产生速度，纯传送
    /// - FixedUpdate 的连续运动仍用 Move* 方法（产生速度，传递接触力）
    /// </summary>
    public void ResetMotion()
    {
        // H1 修复：防御性检查 - 如果 Awake() 还未执行，先初始化
        if (platformRigidbody == null)
        {
            platformRigidbody = GetComponent<Rigidbody>();
            if (platformRigidbody == null)
            {
                platformRigidbody = gameObject.AddComponent<Rigidbody>();
                platformRigidbody.isKinematic = true;
            }
            initialPosition = transform.position;
            initialRotation = transform.rotation;
        }

        // H1: 直接赋值，不通过 Move* 方法（避免产生虚假角速度）
        platformRigidbody.position = initialPosition;
        platformRigidbody.rotation = initialRotation;

        cachedAngularVelocity = Vector3.zero;
        cachedAngularAccel = Vector3.zero;
        cachedLinearAccel = Vector3.zero;

        // H2: 重置 episode 开始时间（触发幅度平滑爬升）
        episodeStartTime = time;
    }

    // ========== Oracle 接口（供预测器消融使用） ==========

    /// <summary>
    /// 获取平台当前欧拉角 (度)
    /// </summary>
    public Vector3 GetPlatformEuler()
    {
        Vector3 euler = platformTransform.rotation.eulerAngles;

        // 归一化到 [-180, 180]
        if (euler.x > 180f) euler.x -= 360f;
        if (euler.y > 180f) euler.y -= 360f;
        if (euler.z > 180f) euler.z -= 360f;

        return euler;
    }

    /// <summary>
    /// 获取平台解析角速度 (deg/s)
    /// </summary>
    public Vector3 GetPlatformAngularVelocity()
    {
        return cachedAngularVelocity;
    }

    /// <summary>
    /// 获取平台解析角加速度 (deg/s²)
    /// </summary>
    public Vector3 GetPlatformAngularAccel()
    {
        return cachedAngularAccel;
    }

    /// <summary>
    /// 获取平台解析线性加速度 (m/s²)
    /// </summary>
    public Vector3 GetPlatformLinearAccel()
    {
        return cachedLinearAccel;
    }
}

/// <summary>
/// 波浪模式枚举
/// </summary>
public enum WaveMode
{
    RegularSine,        // 规则正弦波（向后兼容）
    JONSWAP,            // JONSWAP 谱 (γ=3.3)
    PiersonMoskowitz    // PM 谱 (γ=1.0)
}
