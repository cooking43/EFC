using UnityEngine;
using Unity.MLAgents;

/// <summary>
/// PlatformMotionController - 平台晃动控制器
///
/// 【功能说明】
/// 实现平台的三自由度晃动：
/// - 横摇 (Roll)：绕 Z 轴左右摇摆
/// - 纵摇 (Pitch)：绕 X 轴前后俯仰
/// - 升沉 (Heave)：沿 Y 轴上下起伏
///
/// 【使用步骤】
/// 1. 给 Ground 对象添加 Rigidbody 组件：
///    - Is Kinematic: ✓（必须勾选）
///    - Use Gravity: ✗（不勾选）
///    - Collision Detection: Discrete
///
/// 2. 将此脚本挂载到 Ground 对象上
///
/// 3. 检查碰撞设置：
///    Edit > Project Settings > Physics > Layer Collision Matrix
///    确保 Ground 的 Layer 与机器人的 Layer 之间启用了碰撞
///
/// 4. 初始测试：
///    - 所有 Amplitude 参数保持默认值 0
///    - 先验证添加脚本不影响训练
///    - 确认无影响后，再逐步增加晃动幅度
///
/// 【重要限制】
/// - 横摇/纵摇幅度不能超过 20°（否则机器人会被判定失败）
/// - 建议初始测试使用 5-10° 的幅度
/// - 升沉幅度建议控制在 0.1-0.3m 以内
///
/// 【注意事项】
/// - 必须在 RobotRLAgent 的 OnEpisodeBegin() 中调用 ResetMotion()
/// - 否则每次 Episode 的初始条件会不一致，导致训练不稳定
/// </summary>
public class PlatformMotionController : MonoBehaviour
{
    [Header("横摇参数 (Roll - 左右摇摆)")]
    [Tooltip("横摇幅度（度），0=不晃动。警告：超过20°会导致机器人失败！\n课程学习模式下此值会被 YAML 配置覆盖")]
    [Range(0f, 20f)]
    public float rollAmplitude = 0f;

    [Tooltip("是否启用课程学习（从 YAML 读取幅度）\n消融实验建议设为 false，直接使用 Inspector 参数")]
    public bool useCurriculumLearning = false;

    [Tooltip("横摇频率（Hz），典型值：0.1-0.5Hz")]
    [Range(0.05f, 1f)]
    public float rollFrequency = 0.2f;

    [Tooltip("横摇相位偏移（弧度），用于调整晃动起始位置")]
    public float rollPhase = 0f;

    [Header("纵摇参数 (Pitch - 前后俯仰)")]
    [Tooltip("纵摇幅度（度），0=不晃动。警告：超过20°会导致机器人失败！")]
    [Range(0f, 20f)]
    public float pitchAmplitude = 0f;

    [Tooltip("纵摇频率（Hz），典型值：0.1-0.3Hz")]
    [Range(0.05f, 1f)]
    public float pitchFrequency = 0.15f;

    [Tooltip("纵摇相位偏移（弧度）")]
    public float pitchPhase = 0f;

    [Header("升沉参数 (Heave - 上下起伏)")]
    [Tooltip("升沉幅度（米），0=不晃动")]
    [Range(0f, 0.5f)]
    public float heaveAmplitude = 0f;

    [Tooltip("升沉频率（Hz），典型值：0.1-0.3Hz")]
    [Range(0.05f, 1f)]
    public float heaveFrequency = 0.1f;

    [Tooltip("升沉相位偏移（弧度）")]
    public float heavePhase = 0f;

    [Header("控制")]
    [Tooltip("总开关：是否启用晃动")]
    public bool enableMotion = true;

    [Header("运行时信息 (只读)")]
    [Tooltip("当前欧拉角 (Pitch, Yaw, Roll)")]
    public Vector3 currentEulerAngles;

    [Tooltip("当前位置偏移")]
    public Vector3 currentPositionOffset;

    // 私有变量
    private Rigidbody rb;
    private Vector3 initialPosition;
    private Quaternion initialRotation;
    private float time = 0f;

    void Start()
    {
        // 检查 Rigidbody 组件
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            Debug.LogError($"[PlatformMotionController] {gameObject.name} 缺少 Rigidbody 组件！\n" +
                          "请添加 Rigidbody 并设置 Is Kinematic = true, Use Gravity = false");
            enabled = false;
            return;
        }

        // 检查 Kinematic 设置
        if (!rb.isKinematic)
        {
            Debug.LogWarning($"[PlatformMotionController] {gameObject.name} 的 Rigidbody 应该设为 Kinematic！\n" +
                            "当前设置可能导致平台受重力影响。建议勾选 Is Kinematic。");
        }

        // 检查 Gravity 设置
        if (rb.useGravity)
        {
            Debug.LogWarning($"[PlatformMotionController] {gameObject.name} 的 Rigidbody 不应启用重力！\n" +
                            "建议取消勾选 Use Gravity。");
        }

        // 记录初始状态
        initialPosition = rb.position;
        initialRotation = rb.rotation;

        Debug.Log($"[PlatformMotionController] {gameObject.name} 初始化完成。\n" +
                  $"初始位置: {initialPosition}, 初始旋转: {initialRotation.eulerAngles}\n" +
                  $"当前晃动参数: Roll={rollAmplitude}°, Pitch={pitchAmplitude}°, Heave={heaveAmplitude}m");
    }

    void FixedUpdate()
    {
        // 课程学习：从 ML-Agents 环境参数读取幅度
        float currentRollAmplitude = rollAmplitude;
        if (useCurriculumLearning && Academy.IsInitialized)
        {
            var envParams = Academy.Instance.EnvironmentParameters;
            currentRollAmplitude = envParams.GetWithDefault("platform_roll_amplitude", rollAmplitude);
        }

        // 如果禁用晃动，或所有幅度都为0，直接返回（性能优化）
        if (!enableMotion || (currentRollAmplitude == 0f && pitchAmplitude == 0f && heaveAmplitude == 0f))
        {
            currentEulerAngles = Vector3.zero;
            currentPositionOffset = Vector3.zero;
            return;
        }

        // 更新时间（与物理时间步同步，0.01秒）
        time += Time.fixedDeltaTime;

        // 计算横摇角度 (Roll - 绕 Z 轴旋转)
        // 使用课程学习的幅度值
        float roll = currentRollAmplitude * Mathf.Sin(2f * Mathf.PI * rollFrequency * time + rollPhase);

        // 计算纵摇角度 (Pitch - 绕 X 轴旋转)
        float pitch = pitchAmplitude * Mathf.Sin(2f * Mathf.PI * pitchFrequency * time + pitchPhase);

        // 计算升沉位移 (Heave - Y 轴平移)
        float heave = heaveAmplitude * Mathf.Sin(2f * Mathf.PI * heaveFrequency * time + heavePhase);

        // 应用旋转（欧拉角顺序：Pitch, Yaw=0, Roll）
        Quaternion targetRotation = initialRotation * Quaternion.Euler(pitch, 0, roll);

        // 应用位移（只在 Y 轴上升沉）
        Vector3 targetPosition = initialPosition + new Vector3(0, heave, 0);

        // 使用 Rigidbody 物理方法移动（确保物理引擎能正确计算碰撞和力）
        rb.MoveRotation(targetRotation);
        rb.MovePosition(targetPosition);

        // 更新显示信息
        currentEulerAngles = new Vector3(pitch, 0, roll);
        currentPositionOffset = new Vector3(0, heave, 0);
    }

    /// <summary>
    /// 重置平台到初始状态
    /// 必须在每次 Episode 开始时调用，确保训练的初始条件一致
    /// </summary>
    public void ResetMotion()
    {
        time = 0f;

        if (rb != null)
        {
            rb.position = initialPosition;
            rb.rotation = initialRotation;
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
        }

        currentEulerAngles = Vector3.zero;
        currentPositionOffset = Vector3.zero;

        Debug.Log($"[PlatformMotionController] {gameObject.name} 已重置到初始状态");
    }

    /// <summary>
    /// 获取当前平台的角速度（度/秒）
    /// 可供 Agent 观察使用（未来扩展）
    /// </summary>
    public Vector3 GetAngularVelocity()
    {
        if (rollAmplitude == 0f && pitchAmplitude == 0f)
        {
            return Vector3.zero;
        }

        // 计算角速度的解析解（对时间求导）
        float rollVel = rollAmplitude * 2f * Mathf.PI * rollFrequency *
                        Mathf.Cos(2f * Mathf.PI * rollFrequency * time + rollPhase);

        float pitchVel = pitchAmplitude * 2f * Mathf.PI * pitchFrequency *
                         Mathf.Cos(2f * Mathf.PI * pitchFrequency * time + pitchPhase);

        return new Vector3(pitchVel, 0, rollVel);
    }

    /// <summary>
    /// 获取当前平台的加速度（m/s²）
    /// 可供 Agent 观察使用（未来扩展）
    /// </summary>
    public Vector3 GetAcceleration()
    {
        if (heaveAmplitude == 0f)
        {
            return Vector3.zero;
        }

        // 计算升沉加速度（对时间二阶导）
        float heaveAcc = -heaveAmplitude *
                         Mathf.Pow(2f * Mathf.PI * heaveFrequency, 2) *
                         Mathf.Sin(2f * Mathf.PI * heaveFrequency * time + heavePhase);

        return new Vector3(0, heaveAcc, 0);
    }

    // 在 Scene 视图中绘制调试信息
    void OnDrawGizmos()
    {
        if (!Application.isPlaying) return;

        // 绘制平台中心点
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, 0.5f);

        // 绘制当前旋转的朝向箭头
        Gizmos.color = Color.blue;
        Gizmos.DrawRay(transform.position, transform.forward * 3f);

        // 绘制横摇和纵摇的方向指示
        Gizmos.color = Color.red;
        Gizmos.DrawRay(transform.position, transform.right * 2f);

        Gizmos.color = Color.green;
        Gizmos.DrawRay(transform.position, transform.up * 2f);
    }
}
