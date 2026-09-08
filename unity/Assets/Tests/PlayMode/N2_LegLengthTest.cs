using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// N2: 腿长测试 - 直接判断屈曲/伸展
/// 量髋到踝的距离（腿长）
/// target增加10°前后各测一次:
///   腿长变短 = 屈曲
///   腿长变长 = 伸展
/// </summary>
public class N2_LegLengthTest
{
    [UnityTest]
    public IEnumerator TestKneeFlexionByLegLength()
    {
        Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.None);

#if UNITY_EDITOR
        string scenePath = "Assets/Playground/ShipDeck.unity";
        var loadParams = new UnityEngine.SceneManagement.LoadSceneParameters(UnityEngine.SceneManagement.LoadSceneMode.Single);
        yield return UnityEditor.SceneManagement.EditorSceneManager.LoadSceneAsyncInPlayMode(scenePath, loadParams);
#else
        yield return UnityEngine.SceneManagement.SceneManager.LoadSceneAsync("ShipDeck", UnityEngine.SceneManagement.LoadSceneMode.Single);
#endif

        yield return null;
        yield return null;

        RobotRLAgent agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent");

        Debug.Log("[N2] === 腿长测试（判断屈曲/伸展） ===");

        agent.enabled = false;
        agent.MaxStep = 10000;
        agent.Initialize();
        yield return new WaitForFixedUpdate();

        ArticulationBody robotBase = agent.arts[0];
        robotBase.useGravity = true;

        Debug.Log("[N2] 等待50帧稳定...");
        for (int i = 0; i < 50; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // N2: 测试右膝
        Debug.Log("[N2] --- 测试右膝 ---");
        ArticulationBody rightKnee = agent.acts[agent.IdxRightKnee];
        ArticulationBody rightHipRoll = agent.acts[agent.IdxRightHipRoll];
        ArticulationBody rightAnkleRoll = agent.acts[agent.IdxRightAnkleRoll];

        Assert.IsNotNull(rightKnee, "右膝为null");
        Assert.IsNotNull(rightHipRoll, "右髋为null");
        Assert.IsNotNull(rightAnkleRoll, "右踝为null");

        var rightKneeDrive = rightKnee.xDrive;
        float originalTarget = rightKneeDrive.target;

        // N2: 基线腿长
        Vector3 hipPos = rightHipRoll.transform.position;
        Vector3 anklePos = rightAnkleRoll.transform.position;
        float legLength_baseline = Vector3.Distance(hipPos, anklePos);

        Debug.Log($"[N2] 右膝基线 target: {originalTarget:F2}°");
        Debug.Log($"[N2] 基线腿长: {legLength_baseline:F4} m (髋 {hipPos} → 踝 {anklePos})");

        // N2: 设置target +10°
        rightKneeDrive.target = originalTarget + 10f;
        rightKnee.xDrive = rightKneeDrive;
        Debug.Log($"[N2] 设置右膝 target = {rightKneeDrive.target:F2}° (+10°)");

        LogAssert.Expect(LogType.Error, "[RobotRLAgent] ArticulationBody 引用无效 - 这是严重错误，会导致 episode 静默失败");

        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // N2: 测量新腿长
        hipPos = rightHipRoll.transform.position;
        anklePos = rightAnkleRoll.transform.position;
        float legLength_after = Vector3.Distance(hipPos, anklePos);

        float deltaLegLength = legLength_after - legLength_baseline;

        Debug.Log($"[N2] +10° 后腿长: {legLength_after:F4} m");
        Debug.Log($"[N2] Δ腿长 = {deltaLegLength:F4} m = {deltaLegLength * 1000:F2} mm");

        // 恢复
        rightKneeDrive.target = originalTarget;
        rightKnee.xDrive = rightKneeDrive;

        for (int i = 0; i < 50; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // N2: 测试左膝
        Debug.Log("[N2] --- 测试左膝 ---");
        ArticulationBody leftKnee = agent.acts[agent.IdxLeftKnee];
        ArticulationBody leftHipRoll = agent.acts[agent.IdxLeftHipRoll];
        ArticulationBody leftAnkleRoll = agent.acts[agent.IdxLeftAnkleRoll];

        Assert.IsNotNull(leftKnee, "左膝为null");
        Assert.IsNotNull(leftHipRoll, "左髋为null");
        Assert.IsNotNull(leftAnkleRoll, "左踝为null");

        var leftKneeDrive = leftKnee.xDrive;
        float originalTargetLeft = leftKneeDrive.target;

        // N2: 基线腿长
        hipPos = leftHipRoll.transform.position;
        anklePos = leftAnkleRoll.transform.position;
        float legLength_baselineLeft = Vector3.Distance(hipPos, anklePos);

        Debug.Log($"[N2] 左膝基线 target: {originalTargetLeft:F2}°");
        Debug.Log($"[N2] 基线腿长: {legLength_baselineLeft:F4} m");

        // N2: 设置target +10°
        leftKneeDrive.target = originalTargetLeft + 10f;
        leftKnee.xDrive = leftKneeDrive;
        Debug.Log($"[N2] 设置左膝 target = {leftKneeDrive.target:F2}° (+10°)");

        LogAssert.Expect(LogType.Error, "[RobotRLAgent] ArticulationBody 引用无效 - 这是严重错误，会导致 episode 静默失败");

        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // N2: 测量新腿长
        hipPos = leftHipRoll.transform.position;
        anklePos = leftAnkleRoll.transform.position;
        float legLength_afterLeft = Vector3.Distance(hipPos, anklePos);

        float deltaLegLengthLeft = legLength_afterLeft - legLength_baselineLeft;

        Debug.Log($"[N2] +10° 后腿长: {legLength_afterLeft:F4} m");
        Debug.Log($"[N2] Δ腿长 = {deltaLegLengthLeft:F4} m = {deltaLegLengthLeft * 1000:F2} mm");

        // N2: 输出结论
        Debug.Log("[N2] ===============================");
        Debug.Log("[N2] === 腿长测试结论 ===");
        Debug.Log($"[N2] 右膝 target +10° → 腿长变化 {deltaLegLength * 1000:F2} mm");
        Debug.Log($"[N2] 左膝 target +10° → 腿长变化 {deltaLegLengthLeft * 1000:F2} mm");

        string interpretation = "";
        if (deltaLegLength < -0.001f && deltaLegLengthLeft < -0.001f)
        {
            interpretation = "腿长变短 → target正值 = 屈曲(flexion)";
        }
        else if (deltaLegLength > 0.001f && deltaLegLengthLeft > 0.001f)
        {
            interpretation = "腿长变长 → target正值 = 伸展(extension)";
        }
        else
        {
            interpretation = "腿长变化不显著（<1mm），无法判断";
        }

        Debug.Log($"[N2] 符号约定: {interpretation}");
        Debug.Log("[N2] ===============================");

        Assert.Greater(Mathf.Abs(deltaLegLength), 0.001f, "[N2] 右膝腿长变化应显著 (>1mm)");
        Assert.Greater(Mathf.Abs(deltaLegLengthLeft), 0.001f, "[N2] 左膝腿长变化应显著 (>1mm)");

        Debug.Log("[N2] ✓ 腿长测试完成");
    }
}
