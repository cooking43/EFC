using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// G1: 实测 Unity 中 +roll 的物理方向
///
/// Unity 是左手系、Y 上 Z 前，绕 Z 正转哪边下沉需要实测
/// 不能推理，必须通过代码验证
/// </summary>
public class G1_RollDirectionTest
{
    [UnityTest]
    public IEnumerator TestRollDirection()
    {
        // I5: 禁用 Debug.Log 栈追踪，提高日志可读性
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

        // 查找平台
        RobotRLAgent agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent");

        GameObject platformGO = null;
        Transform areaRoot = agent.transform.parent;

        if (areaRoot != null && areaRoot.GetComponent<TrainingArea>() != null)
        {
            Transform groundTransform = areaRoot.Find("Ground");
            if (groundTransform != null)
                platformGO = groundTransform.gameObject;
        }

        if (platformGO == null)
        {
            platformGO = GameObject.Find("Ground");
        }

        Assert.IsNotNull(platformGO, "未找到平台 GameObject");
        Transform platformTransform = platformGO.transform;

        // I4: 记录平台缩放信息（影响 G2 质心位移计算）
        Debug.Log("[G1] === Unity Roll 方向实测（I2 方向向量法） ===");
        Debug.Log($"[G1] 平台几何信息:");
        Debug.Log($"  localScale: {platformTransform.localScale}");
        Debug.Log($"  lossyScale: {platformTransform.lossyScale}");

        // H3: 禁用平台控制器，防止 FixedUpdate 覆盖旋转
        WaveSpectrumPlatformController controller = platformGO.GetComponent<WaveSpectrumPlatformController>();
        Assert.IsNotNull(controller, "未找到 WaveSpectrumPlatformController");
        Debug.Log("[G1] 禁用平台控制器以确保旋转不被覆盖");
        controller.enabled = false;

        // 获取 Rigidbody
        Rigidbody platformRB = platformGO.GetComponent<Rigidbody>();
        Assert.IsNotNull(platformRB, "未找到平台 Rigidbody");

        const float expectedSin5 = 0.0872f;  // sin(5°)
        const float tolerance = 0.01f;  // 1% tolerance

        // 步骤 1: 平台水平时的基准
        Debug.Log("[G1] --- 步骤 1: 水平基准 ---");
        platformRB.rotation = Quaternion.identity;
        yield return new WaitForFixedUpdate();

        float actualRoll_flat = platformTransform.rotation.eulerAngles.z;
        if (actualRoll_flat > 180f) actualRoll_flat -= 360f;
        Debug.Log($"[G1] 设置: 0°, 实际: {actualRoll_flat:F2}°");
        Assert.IsTrue(Mathf.Abs(actualRoll_flat) < 0.5f, $"水平状态验证失败: {actualRoll_flat:F2}°");

        // I2: 用方向向量法 - 与缩放和轴心无关
        Vector3 rightDir_flat = platformTransform.TransformDirection(Vector3.right);
        Debug.Log($"[G1] 水平时 +X 方向向量: {rightDir_flat}");
        Debug.Log($"  rightDir.y = {rightDir_flat.y:F6}");

        // 步骤 2: 平台 +5° roll
        Debug.Log("[G1] --- 步骤 2: +5° roll ---");
        platformRB.rotation = Quaternion.Euler(0f, 0f, +5f);
        yield return new WaitForFixedUpdate();

        float actualRoll_pos5 = platformTransform.rotation.eulerAngles.z;
        if (actualRoll_pos5 > 180f) actualRoll_pos5 -= 360f;
        Debug.Log($"[G1] 设置: +5°, 实际: {actualRoll_pos5:F2}°");
        Assert.IsTrue(Mathf.Abs(actualRoll_pos5 - 5f) < 0.5f, $"+5° 旋转验证失败: {actualRoll_pos5:F2}°");

        Vector3 rightDir_pos5 = platformTransform.TransformDirection(Vector3.right);
        float rightY_pos5 = rightDir_pos5.y;
        Debug.Log($"[G1] +5° roll 时 +X 方向向量: {rightDir_pos5}");
        Debug.Log($"  rightY = {rightY_pos5:F6} (预期 ≈ {-expectedSin5:F4})");

        // 步骤 3: 平台 -5° roll
        Debug.Log("[G1] --- 步骤 3: -5° roll ---");
        platformRB.rotation = Quaternion.Euler(0f, 0f, -5f);
        yield return new WaitForFixedUpdate();

        float actualRoll_neg5 = platformTransform.rotation.eulerAngles.z;
        if (actualRoll_neg5 > 180f) actualRoll_neg5 -= 360f;
        Debug.Log($"[G1] 设置: -5°, 实际: {actualRoll_neg5:F2}°");
        Assert.IsTrue(Mathf.Abs(actualRoll_neg5 - (-5f)) < 0.5f, $"-5° 旋转验证失败: {actualRoll_neg5:F2}°");

        Vector3 rightDir_neg5 = platformTransform.TransformDirection(Vector3.right);
        float rightY_neg5 = rightDir_neg5.y;
        Debug.Log($"[G1] -5° roll 时 +X 方向向量: {rightDir_neg5}");
        Debug.Log($"  rightY = {rightY_neg5:F6} (预期 ≈ {+expectedSin5:F4})");

        // I2: 三个强制断言（基于方向向量）
        Debug.Log("[G1] === I2 强制断言（方向向量法） ===");

        // 断言 1: |rightY(+5°)| ≈ sin(5°), 容差 1%
        Debug.Log($"[G1] 断言 1: +5° roll 量级检查");
        Debug.Log($"  rightY(+5°) = {rightY_pos5:F6}");
        Debug.Log($"  |rightY(+5°)| = {Mathf.Abs(rightY_pos5):F6}");
        Debug.Log($"  预期 |rightY| = {expectedSin5:F6} ± {tolerance * expectedSin5:F6}");
        Assert.IsTrue(
            Mathf.Abs(Mathf.Abs(rightY_pos5) - expectedSin5) < tolerance * expectedSin5,
            $"|rightY(+5°)| = {Mathf.Abs(rightY_pos5):F6} 应 ≈ sin(5°) = {expectedSin5:F6} ± 1%"
        );
        Debug.Log("  ✓ +5° 量级检查通过");

        // 断言 2: |rightY(-5°)| ≈ sin(5°), 容差 1%
        Debug.Log($"[G1] 断言 2: -5° roll 量级检查");
        Debug.Log($"  rightY(-5°) = {rightY_neg5:F6}");
        Debug.Log($"  |rightY(-5°)| = {Mathf.Abs(rightY_neg5):F6}");
        Debug.Log($"  预期 |rightY| = {expectedSin5:F6} ± {tolerance * expectedSin5:F6}");
        Assert.IsTrue(
            Mathf.Abs(Mathf.Abs(rightY_neg5) - expectedSin5) < tolerance * expectedSin5,
            $"|rightY(-5°)| = {Mathf.Abs(rightY_neg5):F6} 应 ≈ sin(5°) = {expectedSin5:F6} ± 1%"
        );
        Debug.Log("  ✓ -5° 量级检查通过");

        // 断言 3: rightY(+5°) 与 rightY(-5°) 必须反号
        Debug.Log($"[G1] 断言 3: 符号检查");
        Debug.Log($"  rightY(+5°) × rightY(-5°) = {(rightY_pos5 * rightY_neg5):F6}");
        Assert.IsTrue(
            rightY_pos5 * rightY_neg5 < 0,
            $"rightY(+5°) 与 rightY(-5°) 必须反号"
        );
        Debug.Log("  ✓ 符号检查通过");

        // 断言 4: 对称性 - rightY(+5°) + rightY(-5°) ≈ 0
        float sumRightY = rightY_pos5 + rightY_neg5;
        Debug.Log($"[G1] 断言 4: 对称性检查");
        Debug.Log($"  rightY(+5°) + rightY(-5°) = {sumRightY:F6}");
        Debug.Log($"  阈值 = 0.001");
        Assert.IsTrue(
            Mathf.Abs(sumRightY) < 0.001f,
            $"对称性: rightY(+5°) + rightY(-5°) = {sumRightY:F6} 应 < 0.001"
        );
        Debug.Log("  ✓ 对称性检查通过");

        // I3: 明确给出带符号的结论
        Debug.Log("[G1] === Unity Roll 物理约定（实测结论） ===");
        Debug.Log($"[G1] 测量结果（带符号）:");
        Debug.Log($"  rightY(+5°) = {rightY_pos5:F6}");
        Debug.Log($"  rightY(-5°) = {rightY_neg5:F6}");

        string conclusion = "";
        if (rightY_pos5 < 0)
        {
            conclusion = "+roll（绕 Z 轴正向旋转）→ 右侧（+X 方向）下沉";
            Debug.Log($"[G1] ✓ 结论: {conclusion}");
            Debug.Log($"[G1]   物理解释: +roll 使局部 +X 轴向世界 -Y 倾斜");
        }
        else
        {
            conclusion = "+roll（绕 Z 轴正向旋转）→ 左侧（-X 方向）下沉";
            Debug.Log($"[G1] ✓ 结论: {conclusion}");
            Debug.Log($"[G1]   物理解释: +roll 使局部 +X 轴向世界 +Y 倾斜");
        }

        Debug.Log($"[G1] 此结论将用于 G2 判据和前馈控制符号验证");

        // 恢复设置
        controller.enabled = true;
        Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.ScriptOnly);

        yield return null;
    }
}
