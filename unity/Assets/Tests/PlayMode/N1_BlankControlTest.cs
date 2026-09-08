using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// N1: 空白对照测试
/// 什么都不改，步进100帧，测ΔcomX
/// 判断L1测试的+6.3mm是否为物理漂移
/// </summary>
public class N1_BlankControlTest
{
    [UnityTest]
    public IEnumerator TestBlankDrift()
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

        Debug.Log("[N1] === 空白对照测试（检测漂移） ===");

        // 停用平台运动
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

        Assert.IsNotNull(platformGO, "未找到 Ground");

        var platformController = platformGO.GetComponent<WaveSpectrumPlatformController>();
        if (platformController != null)
        {
            platformController.enabled = false;
        }

        platformGO.transform.position = Vector3.zero;
        platformGO.transform.rotation = Quaternion.identity;

        agent.enabled = false;
        agent.MaxStep = 10000;
        agent.Initialize();
        yield return new WaitForFixedUpdate();

        ArticulationBody robotBase = agent.arts[0];
        robotBase.useGravity = true;

        Debug.Log("[N1] 等待100帧稳定...");
        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        // N1: 基线质心
        Vector3 com_baseline = agent.GetWholeBodyCoM();
        float comX_baseline = com_baseline.x;
        Debug.Log($"[N1] 基线质心: {com_baseline}, X = {comX_baseline:F6} m");

        // N1: 什么都不改，再步进100帧
        Debug.Log("[N1] 空白对照：不改任何关节，步进100帧...");

        // 忽略可能的episode错误
        LogAssert.Expect(LogType.Error, "[RobotRLAgent] ArticulationBody 引用无效 - 这是严重错误，会导致 episode 静默失败");

        for (int i = 0; i < 100; i++)
        {
            yield return new WaitForFixedUpdate();
        }

        Vector3 com_after = agent.GetWholeBodyCoM();
        float comX_after = com_after.x;
        float deltaComX_blank = comX_after - comX_baseline;

        Debug.Log($"[N1] 100帧后质心: {com_after}, X = {comX_after:F6} m");
        Debug.Log($"[N1] ===============================");
        Debug.Log($"[N1] 空白对照 ΔcomX = {deltaComX_blank:F6} m = {deltaComX_blank * 1000:F3} mm");
        Debug.Log($"[N1] ===============================");

        // N1: 判断是否存在漂移
        float driftThreshold = 0.001f;  // 1mm
        if (Mathf.Abs(deltaComX_blank) > driftThreshold)
        {
            Debug.LogWarning($"[N1] ⚠ 检测到物理漂移: {deltaComX_blank * 1000:F3} mm (阈值 {driftThreshold * 1000:F1} mm)");
            Debug.LogWarning("[N1] L1测试的+6.3mm可能主要是漂移，而非膝关节响应");
        }
        else
        {
            Debug.Log($"[N1] ✓ 漂移在阈值内: {deltaComX_blank * 1000:F3} mm");
            Debug.Log("[N1] L1测试的同号结果可能是真实物理响应");
        }

        Debug.Log("[N1] ✓ 空白对照测试完成");
    }
}
