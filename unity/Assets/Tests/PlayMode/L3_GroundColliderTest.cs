using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
#if UNITY_EDITOR
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// L3: Ground 碰撞体详细检查（阻塞级）
///
/// 检查 ShipDeck 场景中是否存在地面碰撞体
/// 若无碰撞体，机器人无法站立，之前所有接触测试都要重新审视
/// </summary>
public class L3_GroundColliderTest
{
    [UnityTest]
    public IEnumerator TestGroundColliders()
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

        Debug.Log("[L3] === Ground 碰撞体详细检查 ===");

        // 查找 Agent
        RobotRLAgent agent = Object.FindFirstObjectByType<RobotRLAgent>();
        Assert.IsNotNull(agent, "未找到 RobotRLAgent");

        // 方法1: 通过 TrainingArea 查找 Ground
        GameObject platformGO = null;
        Transform areaRoot = agent.transform.parent;
        if (areaRoot != null && areaRoot.GetComponent<TrainingArea>() != null)
        {
            Debug.Log($"[L3] 找到 TrainingArea: {areaRoot.name}");
            Transform groundTransform = areaRoot.Find("Ground");
            if (groundTransform != null)
            {
                platformGO = groundTransform.gameObject;
                Debug.Log($"[L3] 通过 TrainingArea 找到 Ground: {platformGO.name}");
            }
            else
            {
                Debug.LogWarning("[L3] TrainingArea 下没有名为 'Ground' 的子对象");
            }
        }
        else
        {
            Debug.LogWarning("[L3] Agent 的父对象不是 TrainingArea");
        }

        // 方法2: 全局查找 Ground
        if (platformGO == null)
        {
            platformGO = GameObject.Find("Ground");
            if (platformGO != null)
            {
                Debug.Log($"[L3] 通过全局查找找到 Ground: {platformGO.name}");
            }
        }

        // 方法3: 查找所有包含 "Ground" 或 "Platform" 或 "Floor" 的对象
        Debug.Log("[L3] --- 查找所有疑似地面对象 ---");
        GameObject[] allObjects = Object.FindObjectsByType<GameObject>(FindObjectsSortMode.None);
        int groundLikeCount = 0;
        foreach (GameObject obj in allObjects)
        {
            string nameLower = obj.name.ToLower();
            if (nameLower.Contains("ground") || nameLower.Contains("platform") ||
                nameLower.Contains("floor") || nameLower.Contains("deck"))
            {
                Debug.Log($"[L3] 疑似地面对象: {obj.name} (Layer: {LayerMask.LayerToName(obj.layer)})");
                groundLikeCount++;

                // 检查这个对象及其子对象的 Collider
                Collider[] cols = obj.GetComponentsInChildren<Collider>(true);  // includeInactive = true
                if (cols.Length > 0)
                {
                    foreach (Collider col in cols)
                    {
                        Debug.Log($"  └─ [{col.gameObject.name}] {col.GetType().Name}");
                        Debug.Log($"     Enabled: {col.enabled}, Bounds: {col.bounds.size}");
                        Debug.Log($"     Position: {col.transform.position}");
                    }
                }
                else
                {
                    Debug.LogWarning($"  └─ 无 Collider");
                }
            }
        }
        Debug.Log($"[L3] 共找到 {groundLikeCount} 个疑似地面对象");

        // 详细检查主要的 platformGO
        if (platformGO != null)
        {
            Debug.Log("[L3] --- Ground 对象详细信息 ---");
            Debug.Log($"  名称: {platformGO.name}");
            Debug.Log($"  层级: {LayerMask.LayerToName(platformGO.layer)}");
            Debug.Log($"  激活: {platformGO.activeSelf}");
            Debug.Log($"  世界坐标: {platformGO.transform.position}");
            Debug.Log($"  lossyScale: {platformGO.transform.lossyScale}");

            // 检查所有组件
            Component[] components = platformGO.GetComponents<Component>();
            Debug.Log($"  组件数: {components.Length}");
            foreach (Component comp in components)
            {
                Debug.Log($"    - {comp.GetType().Name}");
            }

            // 检查子对象
            Debug.Log($"  子对象数: {platformGO.transform.childCount}");
            for (int i = 0; i < platformGO.transform.childCount; i++)
            {
                Transform child = platformGO.transform.GetChild(i);
                Debug.Log($"    - {child.name} (激活: {child.gameObject.activeSelf})");
            }

            // 详细检查所有 Collider（包括非激活的）
            Collider[] colliders = platformGO.GetComponentsInChildren<Collider>(true);
            Debug.Log($"  Collider 数（含子对象，含非激活）: {colliders.Length}");

            if (colliders.Length > 0)
            {
                foreach (Collider col in colliders)
                {
                    Debug.Log("[L3] --- Collider 详细信息 ---");
                    Debug.Log($"  对象: {col.gameObject.name}");
                    Debug.Log($"  类型: {col.GetType().Name}");
                    Debug.Log($"  启用: {col.enabled}");
                    Debug.Log($"  isTrigger: {col.isTrigger}");
                    Debug.Log($"  bounds.size: {col.bounds.size}");
                    Debug.Log($"  bounds.center: {col.bounds.center}");
                    Debug.Log($"  世界坐标: {col.transform.position}");

                    // BoxCollider 特殊信息
                    if (col is BoxCollider box)
                    {
                        Debug.Log($"  BoxCollider.size: {box.size}");
                        Debug.Log($"  BoxCollider.center: {box.center}");
                    }
                }

                Assert.IsTrue(colliders.Length > 0, "Ground 应有碰撞体");
            }
            else
            {
                Debug.LogError("[L3] ❌ Ground 及其子对象完全没有 Collider！");
                Assert.Fail("Ground 没有 Collider，机器人无法站立，场景生成有问题");
            }
        }
        else
        {
            Debug.LogError("[L3] ❌ 找不到 Ground 对象！");
            Assert.Fail("ShipDeck 场景中找不到 Ground 对象");
        }

        // 检查机器人与地面的接触
        Debug.Log("[L3] --- 机器人接触检查 ---");
        ArticulationBody[] arts = agent.GetComponentsInChildren<ArticulationBody>();
        int contactCount = 0;
        foreach (ArticulationBody art in arts)
        {
            Collider[] artColliders = art.GetComponents<Collider>();
            foreach (Collider artCol in artColliders)
            {
                if (artCol != null && Physics.ComputePenetration(
                    artCol, artCol.transform.position, artCol.transform.rotation,
                    platformGO.GetComponentInChildren<Collider>(), platformGO.transform.position, platformGO.transform.rotation,
                    out Vector3 direction, out float distance))
                {
                    Debug.Log($"  接触: {art.name} <-> Ground");
                    contactCount++;
                }
            }
        }
        Debug.Log($"[L3] 机器人接触点数: {contactCount}");

        Debug.Log("[L3] === 检查完成 ===");

        // 恢复设置
        Application.SetStackTraceLogType(LogType.Log, StackTraceLogType.ScriptOnly);

        yield return null;
    }
}
