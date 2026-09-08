using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

/// <summary>
/// I2: 测量 Ground 尺寸脚本
///
/// 量三个关键尺寸：
/// a) Ground X/Z 尺寸 - 确保 clone 间距 > 宽度
/// b) Ground Z 长度 - 确保够机器人走完 episode（10s @ 1m/s = 10m）
/// c) 碰撞检测 - 检查 Ground Layer 设置
/// </summary>
public class MeasureGroundDimensions : EditorWindow
{
    [MenuItem("Tools/Measure Ground Dimensions")]
    public static void MeasureDimensions()
    {
        Debug.Log("[MeasureGroundDimensions] 开始测量 Ground 尺寸...");

        // 打开场景
        string scenePath = "Assets/Playground/Playground.unity";
        Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

        if (!scene.IsValid())
        {
            Debug.LogError($"[MeasureGroundDimensions] 无法打开场景: {scenePath}");
            return;
        }

        Debug.Log($"[MeasureGroundDimensions] ✓ 场景已打开: {scene.name}");

        // 查找 Ground 对象（现在应该在 TrainingArea 下）
        GameObject ground = null;

        // 先尝试在 TrainingArea 下查找
        GameObject trainingArea = GameObject.Find("TrainingArea");
        if (trainingArea != null)
        {
            Transform groundTransform = trainingArea.transform.Find("Ground");
            if (groundTransform != null)
            {
                ground = groundTransform.gameObject;
                Debug.Log("[MeasureGroundDimensions] ✓ 在 TrainingArea 下找到 Ground");
            }
        }

        // Fallback: 全局查找
        if (ground == null)
        {
            ground = GameObject.Find("Ground");
            if (ground != null)
            {
                Debug.Log("[MeasureGroundDimensions] ✓ 在场景根级别找到 Ground");
            }
        }

        if (ground == null)
        {
            Debug.LogError("[MeasureGroundDimensions] 未找到 Ground 对象");
            return;
        }

        // 获取尺寸信息
        Debug.Log("\n=== Ground 尺寸测量结果 ===\n");

        // a) Transform scale
        Vector3 scale = ground.transform.localScale;
        Debug.Log($"[I2.a] Ground Transform Scale: X={scale.x:F3}, Y={scale.y:F3}, Z={scale.z:F3}");

        // b) MeshFilter/Renderer bounds (实际占用空间)
        MeshFilter meshFilter = ground.GetComponent<MeshFilter>();
        MeshRenderer meshRenderer = ground.GetComponent<MeshRenderer>();

        if (meshRenderer != null)
        {
            Bounds bounds = meshRenderer.bounds;
            Debug.Log($"[I2.a] Ground Renderer Bounds: X={bounds.size.x:F3}, Y={bounds.size.y:F3}, Z={bounds.size.z:F3}");
            Debug.Log($"[I2.a] Ground Center: X={bounds.center.x:F3}, Y={bounds.center.y:F3}, Z={bounds.center.z:F3}");
            Debug.Log($"[I2.b] Ground Z 长度: {bounds.size.z:F3} m (需要 >= 10m 用于 10s episode @ 1m/s)");

            if (bounds.size.z < 10f)
            {
                Debug.LogWarning($"[I2.b] ⚠ Ground Z 长度 ({bounds.size.z:F3}m) < 10m，可能不够机器人完成 episode");
            }
            else
            {
                Debug.Log($"[I2.b] ✓ Ground Z 长度充足");
            }
        }
        else if (meshFilter != null && meshFilter.sharedMesh != null)
        {
            Bounds localBounds = meshFilter.sharedMesh.bounds;
            Vector3 worldSize = Vector3.Scale(localBounds.size, scale);
            Debug.Log($"[I2.a] Ground Mesh Bounds (scaled): X={worldSize.x:F3}, Y={worldSize.y:F3}, Z={worldSize.z:F3}");
            Debug.Log($"[I2.b] Ground Z 长度: {worldSize.z:F3} m");

            if (worldSize.z < 10f)
            {
                Debug.LogWarning($"[I2.b] ⚠ Ground Z 长度 ({worldSize.z:F3}m) < 10m，可能不够机器人完成 episode");
            }
            else
            {
                Debug.Log($"[I2.b] ✓ Ground Z 长度充足");
            }
        }

        // c) Collider 信息
        Collider[] colliders = ground.GetComponents<Collider>();
        if (colliders.Length > 0)
        {
            Debug.Log($"[I2.c] Ground Collider 数量: {colliders.Length}");
            for (int i = 0; i < colliders.Length; i++)
            {
                Collider col = colliders[i];
                Debug.Log($"  - Collider {i}: {col.GetType().Name}");
                Debug.Log($"    Bounds: X={col.bounds.size.x:F3}, Y={col.bounds.size.y:F3}, Z={col.bounds.size.z:F3}");

                if (col is BoxCollider box)
                {
                    Debug.Log($"    BoxCollider Size: X={box.size.x:F3}, Y={box.size.y:F3}, Z={box.size.z:F3}");
                    Debug.Log($"    BoxCollider Center: X={box.center.x:F3}, Y={box.center.y:F3}, Z={box.center.z:F3}");
                }
            }
        }
        else
        {
            Debug.LogWarning("[I2.c] ⚠ Ground 没有 Collider 组件");
        }

        // d) Layer 设置
        Debug.Log($"[I2.c] Ground Layer: {LayerMask.LayerToName(ground.layer)} (index: {ground.layer})");

        // e) Rigidbody 设置
        Rigidbody rb = ground.GetComponent<Rigidbody>();
        if (rb != null)
        {
            Debug.Log($"[I2.c] Rigidbody: isKinematic={rb.isKinematic}, useGravity={rb.useGravity}");
        }
        else
        {
            Debug.LogWarning("[I2.c] ⚠ Ground 没有 Rigidbody 组件");
        }

        // f) 推荐的 clone 间距
        if (meshRenderer != null)
        {
            float groundWidth = Mathf.Max(meshRenderer.bounds.size.x, meshRenderer.bounds.size.z);
            float recommendedSpacing = groundWidth * 1.5f;  // 1.5x 安全间距
            Debug.Log($"\n[I2 建议] Clone 多个 TrainingArea 时的推荐间距: >= {recommendedSpacing:F3} m ({groundWidth:F3} × 1.5)");
            Debug.Log("[I2 建议] 如果 clone 间距不足，考虑使用不同 Layer 隔离碰撞");
        }

        Debug.Log("\n=== 测量完成 ===\n");
    }
}
