using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// K2/G2b: 关节角到质心的雅可比测试（PlayMode，最小化）
///
/// 测量：右膝+10° 和 左膝+10° 各自对世界系质心 X 的影响
/// 输出：Δcom_x_rightknee 和 Δcom_x_leftknee 的符号
/// </summary>
public class G2b_JacobianTests
{
    [UnityTest]
    public IEnumerator TestKneeJacobian()
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

        // K4: 打印 Ground 尺寸信息
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

        if (platformGO != null)
        {
            Debug.Log("[K4] === Ground 尺寸信息 ===");
            Debug.Log($"  lossyScale: {platformGO.transform.lossyScale}");

            Collider[] colliders = platformGO.GetComponentsInChildren<Collider>();
            if (colliders.Length > 0)
            {
                foreach (Collider col in colliders)
                {
                    Debug.Log($"  [{col.gameObject.name}] Collider.bounds.size: {col.bounds.size}");
                }
            }
            else
            {
                Debug.LogWarning("[K4] Ground 及其子对象无 Collider");
            }
        }

        Debug.Log("[K2/G2b] === 关节角到质心雅可比测试 ===");

        // K2: 停止 Agent 的 FixedUpdate
        agent.enabled = false;
        Debug.Log("[K2] Agent.enabled = false");

        // K2: 机器人悬空，无重力，速度清零
        agent.Initialize();
        yield return new WaitForFixedUpdate();

        ArticulationBody robotBase = agent.arts[0];
        Assert.IsNotNull(robotBase, "机器人骨盆（arts[0]）为 null");

        robotBase.useGravity = false;
        Debug.Log("[K2] robotBase.useGravity = false");

        // 等待初始稳定
        for (int i = 0; i < 10; i++)
        {
            robotBase.velocity = Vector3.zero;
            robotBase.angularVelocity = Vector3.zero;
            yield return new WaitForFixedUpdate();
        }

        // K2: 基线质心
        Vector3 com_baseline = agent.GetWholeBodyCoM();
        float comX_baseline = com_baseline.x;
        Debug.Log($"[K2] 基线质心: {com_baseline}, X = {comX_baseline:F6} m");

        // K2: 测试右膝 +10°
        Debug.Log("[K2] --- 测试右膝 +10° ---");
        ArticulationBody rightKnee = agent.acts[agent.IdxRightKnee];
        Assert.IsNotNull(rightKnee, "右膝关节为 null");

        var rightKneeDrive = rightKnee.xDrive;
        float originalRightTarget = rightKneeDrive.target;
        rightKneeDrive.target = originalRightTarget + 10f;
        rightKnee.xDrive = rightKneeDrive;

        // 步进至稳定
        for (int i = 0; i < 50; i++)
        {
            robotBase.velocity = Vector3.zero;
            robotBase.angularVelocity = Vector3.zero;
            yield return new WaitForFixedUpdate();
        }

        Vector3 com_rightknee = agent.GetWholeBodyCoM();
        float comX_rightknee = com_rightknee.x;
        float deltaComX_rightknee = comX_rightknee - comX_baseline;

        Debug.Log($"[K2] 右膝 +10° 后质心: {com_rightknee}, X = {comX_rightknee:F6} m");
        Debug.Log($"[K2] ΔcomX(右膝) = {deltaComX_rightknee:F6} m");

        // 恢复右膝
        rightKneeDrive.target = originalRightTarget;
        rightKnee.xDrive = rightKneeDrive;

        // 等待恢复
        for (int i = 0; i < 50; i++)
        {
            robotBase.velocity = Vector3.zero;
            robotBase.angularVelocity = Vector3.zero;
            yield return new WaitForFixedUpdate();
        }

        // K2: 测试左膝 +10°
        Debug.Log("[K2] --- 测试左膝 +10° ---");
        ArticulationBody leftKnee = agent.acts[agent.IdxLeftKnee];
        Assert.IsNotNull(leftKnee, "左膝关节为 null");

        var leftKneeDrive = leftKnee.xDrive;
        float originalLeftTarget = leftKneeDrive.target;
        leftKneeDrive.target = originalLeftTarget + 10f;
        leftKnee.xDrive = leftKneeDrive;

        // 步进至稳定
        for (int i = 0; i < 50; i++)
        {
            robotBase.velocity = Vector3.zero;
            robotBase.angularVelocity = Vector3.zero;
            yield return new WaitForFixedUpdate();
        }

        Vector3 com_leftknee = agent.GetWholeBodyCoM();
        float comX_leftknee = com_leftknee.x;
        float deltaComX_leftknee = comX_leftknee - comX_baseline;

        Debug.Log($"[K2] 左膝 +10° 后质心: {com_leftknee}, X = {comX_leftknee:F6} m");
        Debug.Log($"[K2] ΔcomX(左膝) = {deltaComX_leftknee:F6} m");

        // K2: 输出雅可比符号
        Debug.Log("[K2/G2b] === 关节→质心雅可比（符号） ===");
        Debug.Log($"  右膝 +10° → ΔcomX = {deltaComX_rightknee:F6} m (符号: {(deltaComX_rightknee > 0 ? "正" : "负")})");
        Debug.Log($"  左膝 +10° → ΔcomX = {deltaComX_leftknee:F6} m (符号: {(deltaComX_leftknee > 0 ? "正" : "负")})");

        // 断言雅可比应对称且反号
        Assert.IsTrue(deltaComX_rightknee * deltaComX_leftknee < 0,
            $"左右膝雅可比应反号，实际：右膝={deltaComX_rightknee:F6}, 左膝={deltaComX_leftknee:F6}");

        Debug.Log("[K2/G2b] === 测试完成 ===");

        // 恢复设置
        agent.enabled = true;
        robotBase.useGravity = true;
        Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.ScriptOnly);

        yield return null;
    }
}
