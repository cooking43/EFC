using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// L2: 膝关节符号约定测试（PlayMode）
///
/// 目的：确定 ArticulationBody.xDrive.target 和 jointPosition[0] 的符号约定
/// - target +10° → 读取 jointPosition[0] 的变化
/// - 判断 positive = 屈曲(flexion) 还是 伸展(extension)
/// - 记录到代码注释和 PROGRESS.md
/// </summary>
public class L2_KneeSignConventionTests
{
    [UnityTest]
    public IEnumerator TestKneeSignConvention()
    {
        // 禁用 Debug.Log 栈追踪
        Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);

        // 加载 ShipDeck 场景
#if UNITY_EDITOR
        string scenePath = "Assets/Playground/ShipDeck.unity";
        var loadParams = new UnityEngine.SceneManagement.LoadSceneParameters(UnityEngine.SceneManagement.LoadSceneMode.Single);
        yield return UnityEditor.SceneManagement.EditorSceneManager.LoadSceneAsyncInPlayMode(scenePath, loadParams);
#else
        yield return UnityEngine.SceneManagement.SceneManager.LoadSceneAsync("ShipDeck", UnityEngine.SceneManagement.LoadSceneMode.Single);
#endif

        yield return null;
        yield return null;

        // 查找 Agent
        RobotRLAgent agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent");

        Debug.Log("[L2] === 膝关节符号约定测试 ===");

        // L2: 停止 Agent（不需要策略）
        agent.enabled = false;
        Debug.Log("[L2] Agent.enabled = false");

        // L2: 设置 MaxStep 防止 episode 超时
        agent.MaxStep = 10000;
        Debug.Log("[L2] agent.MaxStep = 10000（防止 episode 超时）");

        // L2: 初始化机器人
        agent.Initialize();
        yield return new WaitForFixedUpdate();

        ArticulationBody robotBase = agent.arts[0];
        Assert.IsNotNull(robotBase, "机器人骨盆（arts[0]）为 null");

        // L2: 保持重力（自然站立）
        robotBase.useGravity = true;
        Debug.Log("[L2] robotBase.useGravity = true");

        // L2: 稳定 50 帧
        Debug.Log("[L2] 等待 50 帧物理稳定...");
        for (int i = 0; i < 50; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // L2: 测试右膝
        Debug.Log("[L2] --- 测试右膝符号约定 ---");
        ArticulationBody rightKnee = agent.acts[agent.IdxRightKnee];
        Assert.IsNotNull(rightKnee, "右膝关节为 null");

        // L2: 读取基线
        var rightKneeDrive = rightKnee.xDrive;
        float baselineTarget = rightKneeDrive.target;
        float baselinePosition = rightKnee.jointPosition[0] * Mathf.Rad2Deg;

        Debug.Log($"[L2] 右膝基线 - target: {baselineTarget:F2}°, jointPosition[0]: {baselinePosition:F2}°");

        // L2: 设置 target +10°
        rightKneeDrive.target = baselineTarget + 10f;
        rightKnee.xDrive = rightKneeDrive;
        Debug.Log($"[L2] 设置右膝 target = {rightKneeDrive.target:F2}° (基线 + 10°)");

        // L2: 忽略可能的 ArticulationBody 错误（episode 重置时）
        LogAssert.Expect(LogType.Error, "[RobotRLAgent] ArticulationBody 引用无效 - 这是严重错误，会导致 episode 静默失败");

        // L2: 等待 100 帧稳定
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // L2: 读取新的 jointPosition
        float newPosition = rightKnee.jointPosition[0] * Mathf.Rad2Deg;
        float deltaPosition = newPosition - baselinePosition;

        Debug.Log($"[L2] 右膝新 jointPosition[0]: {newPosition:F2}°");
        Debug.Log($"[L2] ΔjointPosition = {deltaPosition:F2}°");

        // L2: 恢复右膝
        rightKneeDrive.target = baselineTarget;
        rightKnee.xDrive = rightKneeDrive;

        // 等待恢复
        for (int i = 0; i < 50; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // L2: 测试左膝
        Debug.Log("[L2] --- 测试左膝符号约定 ---");
        ArticulationBody leftKnee = agent.acts[agent.IdxLeftKnee];
        Assert.IsNotNull(leftKnee, "左膝关节为 null");

        // L2: 读取基线
        var leftKneeDrive = leftKnee.xDrive;
        float baselineTargetLeft = leftKneeDrive.target;
        float baselinePositionLeft = leftKnee.jointPosition[0] * Mathf.Rad2Deg;

        Debug.Log($"[L2] 左膝基线 - target: {baselineTargetLeft:F2}°, jointPosition[0]: {baselinePositionLeft:F2}°");

        // L2: 设置 target +10°
        leftKneeDrive.target = baselineTargetLeft + 10f;
        leftKnee.xDrive = leftKneeDrive;
        Debug.Log($"[L2] 设置左膝 target = {leftKneeDrive.target:F2}° (基线 + 10°)");

        // L2: 忽略可能的 ArticulationBody 错误（episode 重置时）
        LogAssert.Expect(LogType.Error, "[RobotRLAgent] ArticulationBody 引用无效 - 这是严重错误，会导致 episode 静默失败");

        // L2: 等待 100 帧稳定
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // L2: 读取新的 jointPosition
        float newPositionLeft = leftKnee.jointPosition[0] * Mathf.Rad2Deg;
        float deltaPositionLeft = newPositionLeft - baselinePositionLeft;

        Debug.Log($"[L2] 左膝新 jointPosition[0]: {newPositionLeft:F2}°");
        Debug.Log($"[L2] ΔjointPosition = {deltaPositionLeft:F2}°");

        // L2: 输出符号约定总结
        Debug.Log("[L2] ===============================");
        Debug.Log("[L2] === 膝关节符号约定 ===");
        Debug.Log($"[L2] 右膝: target +10° → jointPosition {(deltaPosition >= 0 ? "+" : "")}{deltaPosition:F2}° ({(deltaPosition >= 0 ? "正值" : "负值")})");
        Debug.Log($"[L2] 左膝: target +10° → jointPosition {(deltaPositionLeft >= 0 ? "+" : "")}{deltaPositionLeft:F2}° ({(deltaPositionLeft >= 0 ? "正值" : "负值")})");

        // L2: 判断符号约定（根据生物力学常识：膝关节正值通常表示屈曲）
        string interpretation = "";
        if (deltaPosition > 0 && deltaPositionLeft > 0)
        {
            interpretation = "正值 = 屈曲(flexion)，负值 = 伸展(extension)";
        }
        else if (deltaPosition < 0 && deltaPositionLeft < 0)
        {
            interpretation = "正值 = 伸展(extension)，负值 = 屈曲(flexion)";
        }
        else
        {
            interpretation = "左右膝符号不一致（需检查关节配置）";
        }

        Debug.Log($"[L2] 符号约定解释: {interpretation}");
        Debug.Log("[L2] ===============================");

        // L2: 断言 target 变化会引起 jointPosition 变化
        Assert.Greater(Mathf.Abs(deltaPosition), 1f,
            "[L2] 右膝 target +10° 应引起 jointPosition 显著变化 (>1°)");
        Assert.Greater(Mathf.Abs(deltaPositionLeft), 1f,
            "[L2] 左膝 target +10° 应引起 jointPosition 显著变化 (>1°)");

        // L2: 断言左右膝符号一致
        Assert.AreEqual(Mathf.Sign(deltaPosition), Mathf.Sign(deltaPositionLeft),
            "[L2] 左右膝的符号约定应一致");

        Debug.Log("[L2] ✓ 膝关节符号约定测试完成");
    }
}
