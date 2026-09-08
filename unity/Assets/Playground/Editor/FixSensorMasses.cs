using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

/// <summary>
/// T1: 修正四个传感器link的质量，从Unity默认1kg改为实际传感器规格值
///
/// 问题：imu_in_pelvis / mid360_link / d435_link / imu_in_torso 都是Unity默认1kg
/// 合计约3.6kg虚构质量（占整机10%），导致全身质心系统性偏移
///
/// 真实质量（来自厂家规格）：
/// - Mid360 (Livox LiDAR): 0.265 kg
/// - RealSense D435: 0.072 kg
/// - IMU (BMI088): 0.005 kg
/// </summary>
public class FixSensorMasses : EditorWindow
{
    [MenuItem("Tools/Fix Sensor Masses")]
    public static void Fix()
    {
        // 打开Playground场景
        var scene = EditorSceneManager.OpenScene("Assets/Playground/Playground.unity");

        // 传感器真实质量配置
        var sensorMasses = new System.Collections.Generic.Dictionary<string, float>
        {
            {"mid360_link", 0.265f},      // Livox Mid-360 LiDAR
            {"d435_link", 0.072f},         // Intel RealSense D435
            {"imu_in_pelvis", 0.005f},     // BMI088 IMU
            {"imu_in_torso", 0.005f}       // BMI088 IMU
        };

        int fixedCount = 0;
        float totalReduction = 0f;

        // 查找并更新每个传感器link
        foreach (var kvp in sensorMasses)
        {
            GameObject[] allObjects = scene.GetRootGameObjects();
            ArticulationBody targetBody = null;

            // 递归查找
            foreach (var root in allObjects)
            {
                targetBody = FindArticulationBodyRecursive(root.transform, kvp.Key);
                if (targetBody != null) break;
            }

            if (targetBody != null)
            {
                float oldMass = targetBody.mass;
                targetBody.mass = kvp.Value;
                totalReduction += (oldMass - kvp.Value);
                fixedCount++;

                Debug.Log($"[FixSensorMasses] {kvp.Key}: {oldMass:F3} kg → {kvp.Value:F3} kg (减少 {oldMass - kvp.Value:F3} kg)");
            }
            else
            {
                Debug.LogWarning($"[FixSensorMasses] 未找到: {kvp.Key}");
            }
        }

        // 保存场景
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);

        Debug.Log($"[FixSensorMasses] 完成！修正了 {fixedCount}/4 个传感器");
        Debug.Log($"[FixSensorMasses] 总减少虚构质量: {totalReduction:F3} kg ({totalReduction/35f*100:F1}% of 35kg nominal)");
        Debug.Log($"[FixSensorMasses] 场景已保存: {scene.path}");

        // 刷新
        AssetDatabase.Refresh();
    }

    private static ArticulationBody FindArticulationBodyRecursive(Transform parent, string targetName)
    {
        if (parent.name == targetName)
        {
            return parent.GetComponent<ArticulationBody>();
        }

        foreach (Transform child in parent)
        {
            var result = FindArticulationBodyRecursive(child, targetName);
            if (result != null) return result;
        }

        return null;
    }
}
