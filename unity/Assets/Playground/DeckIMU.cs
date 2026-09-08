using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// M1: 甲板 IMU - 真前馈传感器
///
/// 独立测量平台运动状态，为预测器提供真正的前馈信号
/// （区别于从机器人自身状态推导，后者是超前校正而非前馈）
///
/// 功能：
/// - Oracle 模式：从 WaveSpectrumPlatformController 读取解析真值
/// - Sensor 模式：基于平台 Rigidbody 状态 + 可配置噪声/延迟
/// - 输出：平台位置、速度、角速度、姿态
/// </summary>
public class DeckIMU : MonoBehaviour
{
    public enum SensorMode
    {
        Oracle,   // 解析真值（性能上界）
        Sensor    // 仿真 IMU（噪声 + 延迟）
    }

    [Header("传感器模式")]
    [Tooltip("Oracle: 解析真值 | Sensor: 仿真 IMU")]
    public SensorMode mode = SensorMode.Oracle;

    [Header("噪声参数 (Sensor模式)")]
    [Tooltip("角速度噪声标准差 (rad/s)")]
    public float angularVelocityNoise = 0.001f;  // ~0.06°/s

    [Tooltip("加速度噪声标准差 (m/s²)")]
    public float accelerationNoise = 0.01f;

    [Tooltip("姿态噪声标准差 (degrees)")]
    public float attitudeNoise = 0.1f;

    [Header("延迟参数 (Sensor模式)")]
    [Tooltip("传感器延迟 (秒)")]
    [Range(0f, 0.1f)]
    public float sensorDelay = 0.01f;  // 10ms

    [Header("平台引用")]
    [Tooltip("平台对象（自动查找或手动指定）")]
    public GameObject platformObject;

    [Header("调试输出")]
    public Vector3 currentPlatformVelocity;
    public Vector3 currentPlatformAngularVelocity;
    public Vector3 currentPlatformEulerAngles;
    public Vector3 currentPlatformPosition;

    // 内部状态
    private Rigidbody platformRigidbody;
    private WaveSpectrumPlatformController platformController;
    private Queue<PlatformState> delayBuffer;

    /// <summary>
    /// 平台状态记录（用于延迟缓冲）
    /// </summary>
    private class PlatformState
    {
        public Vector3 position;
        public Vector3 velocity;
        public Vector3 angularVelocity;
        public Vector3 eulerAngles;
        public float timestamp;

        public PlatformState(Vector3 pos, Vector3 vel, Vector3 angVel, Vector3 euler, float time)
        {
            position = pos;
            velocity = vel;
            angularVelocity = angVel;
            eulerAngles = euler;
            timestamp = time;
        }
    }

    void Awake()
    {
        delayBuffer = new Queue<PlatformState>();

        // 自动查找平台对象
        if (platformObject == null)
        {
            // 尝试查找 Ground 对象
            platformObject = GameObject.Find("Ground");

            // 如果在 TrainingArea 中，查找相对路径
            if (platformObject == null)
            {
                Transform areaRoot = transform.parent;
                if (areaRoot != null && areaRoot.GetComponent<TrainingArea>() != null)
                {
                    Transform groundTransform = areaRoot.Find("Ground");
                    if (groundTransform != null)
                        platformObject = groundTransform.gameObject;
                }
            }

            if (platformObject == null)
            {
                Debug.LogError("[DeckIMU] 无法找到平台对象 (Ground)！");
                return;
            }

            Debug.Log($"[DeckIMU] 自动找到平台: {platformObject.name}");
        }

        // 获取平台组件
        platformRigidbody = platformObject.GetComponent<Rigidbody>();
        platformController = platformObject.GetComponent<WaveSpectrumPlatformController>();

        if (platformRigidbody == null)
        {
            Debug.LogWarning($"[DeckIMU] 平台 {platformObject.name} 没有 Rigidbody（可能是运动学平台）");
        }

        if (platformController == null)
        {
            Debug.LogWarning($"[DeckIMU] 平台 {platformObject.name} 没有 WaveSpectrumPlatformController（Oracle 模式不可用）");
        }

        Debug.Log($"[DeckIMU] 初始化完成 - 模式: {mode}, " +
                 $"延迟: {sensorDelay * 1000:F1}ms, " +
                 $"姿态噪声: {attitudeNoise:F3}°");
    }

    void FixedUpdate()
    {
        // 读取平台真实状态
        Vector3 truePosition = platformObject.transform.position;
        Vector3 trueEuler = platformObject.transform.eulerAngles;

        Vector3 trueVelocity = Vector3.zero;
        Vector3 trueAngularVelocity = Vector3.zero;

        if (mode == SensorMode.Oracle && platformController != null)
        {
            // Oracle 模式：从 WaveSpectrumPlatformController 获取解析真值
            // 注意：GetPlatformAngularVelocity() 返回 deg/s，需要转换为 rad/s
            Vector3 angVelDegPerSec = platformController.GetPlatformAngularVelocity();
            trueAngularVelocity = angVelDegPerSec * Mathf.Deg2Rad;

            // 速度可以从 Rigidbody 获取（如果有），否则从角速度推导
            if (platformRigidbody != null)
            {
                trueVelocity = platformRigidbody.velocity;
            }
        }
        else if (platformRigidbody != null)
        {
            // Sensor 模式或 Oracle 不可用：从 Rigidbody 读取
            trueVelocity = platformRigidbody.velocity;
            trueAngularVelocity = platformRigidbody.angularVelocity;
        }

        // 添加噪声（Sensor 模式）
        if (mode == SensorMode.Sensor)
        {
            trueVelocity += new Vector3(
                GaussianNoise() * accelerationNoise * Time.fixedDeltaTime,
                GaussianNoise() * accelerationNoise * Time.fixedDeltaTime,
                GaussianNoise() * accelerationNoise * Time.fixedDeltaTime
            );

            trueAngularVelocity += new Vector3(
                GaussianNoise() * angularVelocityNoise,
                GaussianNoise() * angularVelocityNoise,
                GaussianNoise() * angularVelocityNoise
            );

            trueEuler += new Vector3(
                GaussianNoise() * attitudeNoise,
                GaussianNoise() * attitudeNoise,
                GaussianNoise() * attitudeNoise
            );
        }

        // 加入延迟缓冲
        PlatformState currentState = new PlatformState(
            truePosition, trueVelocity, trueAngularVelocity, trueEuler, Time.fixedTime
        );
        delayBuffer.Enqueue(currentState);

        // 移除超过延迟时间的旧状态
        while (delayBuffer.Count > 0 &&
               Time.fixedTime - delayBuffer.Peek().timestamp > sensorDelay)
        {
            delayBuffer.Dequeue();
        }

        // 输出延迟后的状态
        if (delayBuffer.Count > 0)
        {
            PlatformState delayedState = delayBuffer.Peek();
            currentPlatformPosition = delayedState.position;
            currentPlatformVelocity = delayedState.velocity;
            currentPlatformAngularVelocity = delayedState.angularVelocity;
            currentPlatformEulerAngles = delayedState.eulerAngles;
        }
        else
        {
            // 缓冲未满，使用当前状态
            currentPlatformPosition = truePosition;
            currentPlatformVelocity = trueVelocity;
            currentPlatformAngularVelocity = trueAngularVelocity;
            currentPlatformEulerAngles = trueEuler;
        }
    }

    /// <summary>
    /// 获取平台状态（供 ExplicitVelocityPredictor 调用）
    /// </summary>
    public void GetPlatformState(out Vector3 velocity, out Vector3 angularVelocity,
                                  out Vector3 position, out Vector3 eulerAngles)
    {
        velocity = currentPlatformVelocity;
        angularVelocity = currentPlatformAngularVelocity;
        position = currentPlatformPosition;
        eulerAngles = currentPlatformEulerAngles;
    }

    /// <summary>
    /// Box-Muller 高斯噪声生成
    /// </summary>
    private float GaussianNoise()
    {
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    /// <summary>
    /// 调试可视化
    /// </summary>
    void OnDrawGizmos()
    {
        if (!Application.isPlaying || platformObject == null)
            return;

        // 绘制平台角速度（洋红色）
        Gizmos.color = Color.magenta;
        Vector3 angVelViz = currentPlatformAngularVelocity * Mathf.Rad2Deg;  // 转换为deg/s用于可视化
        Gizmos.DrawRay(platformObject.transform.position, angVelViz * 0.1f);

        // 绘制平台速度（青色）
        Gizmos.color = Color.cyan;
        Gizmos.DrawRay(platformObject.transform.position, currentPlatformVelocity);
    }
}
