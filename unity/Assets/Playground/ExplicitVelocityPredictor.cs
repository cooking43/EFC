using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 显式速度预测器
/// 基于泰勒展开的速度预测理论，显式计算速度、加速度和未来状态预测
///
/// 理论基础：
/// v(t + Δt) ≈ v(t) + v'(t)·Δt + v''(t)·Δt²/2
///
/// 其中：
/// - v'(t) = dv/dt（一阶导数，加速度）
/// - v''(t) = d²v/dt²（二阶导数，jerk）
///
/// 应用场景：
/// 用于预测机器人或平台的未来运动状态，使策略能够主动适应而非被动响应
/// </summary>
public class ExplicitVelocityPredictor : MonoBehaviour
{
    /// <summary>
    /// 历史状态缓存
    /// </summary>
    private class StateHistory
    {
        public Vector3 velocity;
        public Vector3 angularVelocity;
        public Vector3 position;
        public Vector3 eulerAngles;
        public float timestamp;

        public StateHistory(Vector3 vel, Vector3 angVel, Vector3 pos, Vector3 euler, float time)
        {
            velocity = vel;
            angularVelocity = angVel;
            position = pos;
            eulerAngles = euler;
            timestamp = time;
        }
    }

    private Queue<StateHistory> historyBuffer = new Queue<StateHistory>();
    private int maxHistorySize = 3;  // 3帧用于二阶泰勒展开

    [Header("预测参数")]
    [Tooltip("预测的时间步长（秒），建议0.1-0.3秒")]
    public float predictionHorizon = 0.1f;

    [Header("调试信息")]
    public Vector3 currentVelocity;
    public Vector3 velocityFirstDerivative;   // 一阶导数（加速度）
    public Vector3 velocitySecondDerivative;  // 二阶导数（jerk）
    public Vector3 predictedVelocity;

    public Vector3 currentAngularVelocity;
    public Vector3 angularVelocityFirstDerivative;
    public Vector3 angularVelocitySecondDerivative;
    public Vector3 predictedAngularVelocity;

    public Vector3 predictedEulerAngles;

    /// <summary>
    /// 更新历史状态
    /// 每个FixedUpdate调用一次
    /// </summary>
    public void UpdateHistory(Vector3 velocity, Vector3 angularVelocity, Vector3 position, Vector3 eulerAngles)
    {
        float currentTime = Time.fixedTime;

        // 添加新状态
        historyBuffer.Enqueue(new StateHistory(velocity, angularVelocity, position, eulerAngles, currentTime));

        // 保持固定大小
        while (historyBuffer.Count > maxHistorySize)
        {
            historyBuffer.Dequeue();
        }
    }

    /// <summary>
    /// 计算速度的一阶导数（加速度）
    /// 使用有限差分法：v'(t) ≈ (v_t - v_{t-1}) / Δt
    /// </summary>
    private Vector3 ComputeFirstDerivative(List<StateHistory> history, System.Func<StateHistory, Vector3> selector)
    {
        if (history.Count < 2)
            return Vector3.zero;

        int n = history.Count;
        Vector3 v_t = selector(history[n - 1]);
        Vector3 v_t1 = selector(history[n - 2]);
        float dt = history[n - 1].timestamp - history[n - 2].timestamp;

        if (dt <= 0.0001f)
            return Vector3.zero;

        return (v_t - v_t1) / dt;
    }

    /// <summary>
    /// 计算速度的二阶导数（jerk）
    /// 使用有限差分法：v''(t) ≈ (v_t - 2v_{t-1} + v_{t-2}) / Δt²
    /// </summary>
    private Vector3 ComputeSecondDerivative(List<StateHistory> history, System.Func<StateHistory, Vector3> selector)
    {
        if (history.Count < 3)
            return Vector3.zero;

        int n = history.Count;
        Vector3 v_t = selector(history[n - 1]);
        Vector3 v_t1 = selector(history[n - 2]);
        Vector3 v_t2 = selector(history[n - 3]);
        float dt = history[n - 1].timestamp - history[n - 2].timestamp;

        if (dt <= 0.0001f)
            return Vector3.zero;

        return (v_t - 2f * v_t1 + v_t2) / (dt * dt);
    }

    /// <summary>
    /// 使用二阶泰勒展开预测未来速度
    /// v(t + Δt) ≈ v(t) + v'(t)·Δt + v''(t)·Δt²/2
    /// </summary>
    private Vector3 PredictFutureValue(Vector3 current, Vector3 firstDerivative, Vector3 secondDerivative, float dt)
    {
        return current + firstDerivative * dt + 0.5f * secondDerivative * dt * dt;
    }

    /// <summary>
    /// 预测未来的欧拉角
    /// θ(t + Δt) = θ(t) + ω(t)·Δt + ω'(t)·Δt²/2
    /// </summary>
    private Vector3 PredictFutureEulerAngles(Vector3 currentEuler, Vector3 angVel, Vector3 angAcc, float dt)
    {
        // 将角速度和角加速度转换为度/秒
        Vector3 angVelDegrees = angVel * Mathf.Rad2Deg;
        Vector3 angAccDegrees = angAcc * Mathf.Rad2Deg;

        // 预测角度变化
        Vector3 predictedEuler = currentEuler + angVelDegrees * dt + 0.5f * angAccDegrees * dt * dt;

        // 归一化到[-180, 180]
        predictedEuler.x = NormalizeAngle(predictedEuler.x);
        predictedEuler.y = NormalizeAngle(predictedEuler.y);
        predictedEuler.z = NormalizeAngle(predictedEuler.z);

        return predictedEuler;
    }

    /// <summary>
    /// 将角度归一化到[-180, 180]
    /// </summary>
    private float NormalizeAngle(float angle)
    {
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    /// <summary>
    /// 获取完整的预测观测向量
    /// 返回：当前状态 + 一阶导数 + 二阶导数 + 预测值
    /// </summary>
    public void GetPredictionObservations(out Vector3[] observations)
    {
        List<Vector3> obs = new List<Vector3>();

        if (historyBuffer.Count < 3)
        {
            // 历史不足，返回零向量
            observations = new Vector3[12];
            for (int i = 0; i < 12; i++)
                observations[i] = Vector3.zero;
            return;
        }

        List<StateHistory> history = new List<StateHistory>(historyBuffer);

        // 1. 当前线速度
        currentVelocity = history[history.Count - 1].velocity;
        obs.Add(currentVelocity);

        // 2. 线速度一阶导数（加速度）
        velocityFirstDerivative = ComputeFirstDerivative(history, s => s.velocity);
        obs.Add(velocityFirstDerivative);

        // 3. 线速度二阶导数（jerk）
        velocitySecondDerivative = ComputeSecondDerivative(history, s => s.velocity);
        obs.Add(velocitySecondDerivative);

        // 4. 预测的未来线速度
        predictedVelocity = PredictFutureValue(currentVelocity, velocityFirstDerivative,
                                                velocitySecondDerivative, predictionHorizon);
        obs.Add(predictedVelocity);

        // 5. 当前角速度
        currentAngularVelocity = history[history.Count - 1].angularVelocity;
        obs.Add(currentAngularVelocity);

        // 6. 角速度一阶导数（角加速度）
        angularVelocityFirstDerivative = ComputeFirstDerivative(history, s => s.angularVelocity);
        obs.Add(angularVelocityFirstDerivative);

        // 7. 角速度二阶导数
        angularVelocitySecondDerivative = ComputeSecondDerivative(history, s => s.angularVelocity);
        obs.Add(angularVelocitySecondDerivative);

        // 8. 预测的未来角速度
        predictedAngularVelocity = PredictFutureValue(currentAngularVelocity,
                                                       angularVelocityFirstDerivative,
                                                       angularVelocitySecondDerivative,
                                                       predictionHorizon);
        obs.Add(predictedAngularVelocity);

        // 9. 当前欧拉角
        Vector3 currentEuler = history[history.Count - 1].eulerAngles;
        obs.Add(currentEuler);

        // 10. 预测的未来欧拉角
        predictedEulerAngles = PredictFutureEulerAngles(currentEuler,
                                                        currentAngularVelocity,
                                                        angularVelocityFirstDerivative,
                                                        predictionHorizon);
        obs.Add(predictedEulerAngles);

        observations = obs.ToArray();
    }

    /// <summary>
    /// 获取简化版本的预测观测（仅包含预测值，不包含导数）
    /// 返回：预测的速度 + 预测的角速度 + 预测的姿态
    /// </summary>
    public void GetSimplifiedPredictionObservations(out Vector3 predVel, out Vector3 predAngVel, out Vector3 predEuler)
    {
        if (historyBuffer.Count < 3)
        {
            predVel = Vector3.zero;
            predAngVel = Vector3.zero;
            predEuler = Vector3.zero;
            return;
        }

        List<StateHistory> history = new List<StateHistory>(historyBuffer);

        // 计算线速度预测
        currentVelocity = history[history.Count - 1].velocity;
        velocityFirstDerivative = ComputeFirstDerivative(history, s => s.velocity);
        velocitySecondDerivative = ComputeSecondDerivative(history, s => s.velocity);
        predVel = PredictFutureValue(currentVelocity, velocityFirstDerivative,
                                      velocitySecondDerivative, predictionHorizon);

        // 计算角速度预测
        currentAngularVelocity = history[history.Count - 1].angularVelocity;
        angularVelocityFirstDerivative = ComputeFirstDerivative(history, s => s.angularVelocity);
        angularVelocitySecondDerivative = ComputeSecondDerivative(history, s => s.angularVelocity);
        predAngVel = PredictFutureValue(currentAngularVelocity, angularVelocityFirstDerivative,
                                         angularVelocitySecondDerivative, predictionHorizon);

        // 计算姿态预测
        Vector3 currentEuler = history[history.Count - 1].eulerAngles;
        predEuler = PredictFutureEulerAngles(currentEuler, currentAngularVelocity,
                                             angularVelocityFirstDerivative, predictionHorizon);
    }

    /// <summary>
    /// 在Scene视图中可视化预测信息
    /// </summary>
    void OnDrawGizmos()
    {
        if (!Application.isPlaying || historyBuffer.Count < 3)
            return;

        // 绘制当前速度箭头（蓝色）
        Gizmos.color = Color.blue;
        Gizmos.DrawRay(transform.position, currentVelocity);

        // 绘制预测速度箭头（绿色）
        Gizmos.color = Color.green;
        Gizmos.DrawRay(transform.position, predictedVelocity);

        // 绘制加速度方向（红色）
        Gizmos.color = Color.red;
        Gizmos.DrawRay(transform.position, velocityFirstDerivative * 0.1f);
    }
}
