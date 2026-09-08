using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// L1: 站立雅可比测试（PlayMode，双支撑着地）
///
/// 测量：机器人站在水平甲板上，双支撑状态
/// 右膝+10° 和 左膝+10° 各自对世界系质心 X 的影响
/// 输出：Δcom_x_rightknee 和 Δcom_x_leftknee 的符号
///
/// 与 K2 的差异：K2 悬空无重力（移除了膝关节影响侧向平衡的机制）
/// L1 站地有重力（支撑腿改变腿长 → 骨盆倾斜 → 质心侧移）
/// </summary>
public class L1_StandingJacobianTests
{
    [UnityTest]
    public IEnumerator TestKneeJacobianStanding()
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

        Debug.Log("[L1] === 站立雅可比测试（双支撑） ===");

        // L1: 查找平台控制器，确保平台水平不动
        GameObject platformGO = null;
        Transform areaRoot = agent.transform.parent;
        if (areaRoot != null && areaRoot.GetComponent<TrainingArea>() != null)
        {
            Transform groundTransform = areaRoot.Find("Ground");
            if (groundTransform != null)
                platformGO = groundTransform.gameObject;
        }
        if (platformGO == null)
            platformGO = GameObject.Find("Ground");

        Assert.IsNotNull(platformGO, "未找到 Ground 对象");

        // L1: 停止平台运动（保持水平）
        var platformController = platformGO.GetComponent<WaveSpectrumPlatformController>();
        if (platformController != null)
        {
            platformController.enabled = false;
            Debug.Log("[L1] 平台控制器已停用（保持水平）");
        }

        // 确保平台位姿水平
        platformGO.transform.position = Vector3.zero;
        platformGO.transform.rotation = Quaternion.identity;

        // 验证 Ground 有 Collider
        Collider[] colliders = platformGO.GetComponentsInChildren<Collider>();
        Assert.Greater(colliders.Length, 0, "[L1] Ground 必须有 Collider 才能测试站立雅可比");
        Debug.Log($"[L1] Ground 有 {colliders.Length} 个 Collider: {string.Join(", ", System.Array.ConvertAll(colliders, c => c.GetType().Name))}");

        // L1: 停止 Agent 的 FixedUpdate（不需要策略网络）
        agent.enabled = false;
        Debug.Log("[L1] Agent.enabled = false");

        // L1: 初始化机器人（重要：保持重力 ON，让机器人站立）
        agent.Initialize();
        yield return new WaitForFixedUpdate();

        ArticulationBody robotBase = agent.arts[0];
        Assert.IsNotNull(robotBase, "机器人骨盆（arts[0]）为 null");

        // L1: 确保重力 ON（与 K2 的关键区别）
        robotBase.useGravity = true;
        Debug.Log("[L1] robotBase.useGravity = true（机器人站地）");

        // L1: 等待 100 帧稳定（双支撑站立）
        Debug.Log("[L1] 等待 100 帧物理稳定...");
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // L1: 打印机器人初始状态
        float robotY = robotBase.transform.position.y;
        float groundY = platformGO.transform.position.y;
        Debug.Log($"[L1] 稳定后：机器人 Y={robotY:F3}m, Ground Y={groundY:F3}m, 高度差={robotY - groundY:F3}m");

        // 验证机器人在地面上方（未穿地）
        Assert.Greater(robotY, groundY, "[L1] 机器人必须在 Ground 上方（站立状态）");

        // L1: 基线质心
        Vector3 com_baseline = agent.GetWholeBodyCoM();
        float comX_baseline = com_baseline.x;
        Debug.Log($"[L1] 基线质心: {com_baseline}, X = {comX_baseline:F6} m");

        // L1: 测试右膝 +10°
        Debug.Log("[L1] --- 测试右膝 +10° ---");
        ArticulationBody rightKnee = agent.acts[agent.IdxRightKnee];
        Assert.IsNotNull(rightKnee, "右膝关节为 null");

        var rightKneeDrive = rightKnee.xDrive;
        float originalRightTarget = rightKneeDrive.target;
        Debug.Log($"[L1] 右膝原始 target: {originalRightTarget:F2}°");

        rightKneeDrive.target = originalRightTarget + 10f;
        rightKnee.xDrive = rightKneeDrive;
        Debug.Log($"[L1] 右膝新 target: {rightKneeDrive.target:F2}°");

        // L1: 步进 100 帧至稳定
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        Vector3 com_rightknee = agent.GetWholeBodyCoM();
        float comX_rightknee = com_rightknee.x;
        float deltaComX_rightknee = comX_rightknee - comX_baseline;

        Debug.Log($"[L1] 右膝 +10° 后质心: {com_rightknee}, X = {comX_rightknee:F6} m");
        Debug.Log($"[L1] ΔcomX(右膝) = {deltaComX_rightknee:F6} m = {deltaComX_rightknee * 1000:F3} mm");

        // 恢复右膝
        rightKneeDrive.target = originalRightTarget;
        rightKnee.xDrive = rightKneeDrive;
        Debug.Log($"[L1] 恢复右膝 target: {originalRightTarget:F2}°");

        // 等待恢复稳定
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // L1: 测试左膝 +10°
        Debug.Log("[L1] --- 测试左膝 +10° ---");
        ArticulationBody leftKnee = agent.acts[agent.IdxLeftKnee];
        Assert.IsNotNull(leftKnee, "左膝关节为 null");

        var leftKneeDrive = leftKnee.xDrive;
        float originalLeftTarget = leftKneeDrive.target;
        Debug.Log($"[L1] 左膝原始 target: {originalLeftTarget:F2}°");

        leftKneeDrive.target = originalLeftTarget + 10f;
        leftKnee.xDrive = leftKneeDrive;
        Debug.Log($"[L1] 左膝新 target: {leftKneeDrive.target:F2}°");

        // L1: 步进 100 帧至稳定
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        Vector3 com_leftknee = agent.GetWholeBodyCoM();
        float comX_leftknee = com_leftknee.x;
        float deltaComX_leftknee = comX_leftknee - comX_baseline;

        Debug.Log($"[L1] 左膝 +10° 后质心: {com_leftknee}, X = {comX_leftknee:F6} m");
        Debug.Log($"[L1] ΔcomX(左膝) = {deltaComX_leftknee:F6} m = {deltaComX_leftknee * 1000:F3} mm");

        // L1: 输出雅可比符号总结
        Debug.Log("[L1] ===============================");
        Debug.Log("[L1] === 雅可比符号测量结果 ===");
        Debug.Log($"[L1] 右膝 +10° → ΔcomX = {deltaComX_rightknee * 1000:F3} mm ({(deltaComX_rightknee >= 0 ? "正" : "负")})");
        Debug.Log($"[L1] 左膝 +10° → ΔcomX = {deltaComX_leftknee * 1000:F3} mm ({(deltaComX_leftknee >= 0 ? "正" : "负")})");
        Debug.Log("[L1] ===============================");

        // L1: 断言雅可比不为零（与 K2 悬空结果不同）
        Assert.Greater(Mathf.Abs(deltaComX_rightknee), 0.001f,
            "[L1] 右膝雅可比不应为零（站立状态下膝关节影响侧向平衡）");
        Assert.Greater(Mathf.Abs(deltaComX_leftknee), 0.001f,
            "[L1] 左膝雅可比不应为零（站立状态下膝关节影响侧向平衡）");

        Debug.Log("[L1] ✓ 站立雅可比测试完成");
    }
}
