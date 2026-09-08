using UnityEngine;
using UnityEditor;

/// <summary>
/// L2: 将 ShipDeck.unity 加入 Build Settings（headless build 需要）
/// </summary>
public static class ConfigureBuildSettings
{
    [MenuItem("Tools/Configure Build Settings")]
    public static void AddShipDeckToBuildSettings()
    {
        Debug.Log("[ConfigureBuildSettings] 开始配置 Build Settings...");

        string shipDeckPath = "Assets/Playground/ShipDeck.unity";

        // 获取当前 Build Settings 中的场景列表
        var scenes = new System.Collections.Generic.List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);

        // 检查 ShipDeck.unity 是否已存在
        bool alreadyExists = false;
        foreach (var scene in scenes)
        {
            if (scene.path == shipDeckPath)
            {
                alreadyExists = true;
                if (scene.enabled)
                {
                    Debug.Log($"[ConfigureBuildSettings] ✓ {shipDeckPath} 已在 Build Settings 中且已启用");
                }
                else
                {
                    scene.enabled = true;
                    Debug.Log($"[ConfigureBuildSettings] ✓ {shipDeckPath} 已在 Build Settings 中，已启用");
                }
                break;
            }
        }

        // 如果不存在，添加到列表
        if (!alreadyExists)
        {
            scenes.Add(new EditorBuildSettingsScene(shipDeckPath, true));
            Debug.Log($"[ConfigureBuildSettings] ✓ 已添加 {shipDeckPath} 到 Build Settings");
        }

        // 更新 Build Settings
        EditorBuildSettings.scenes = scenes.ToArray();

        Debug.Log("[ConfigureBuildSettings] ✓✓✓ Build Settings 配置完成");
        Debug.Log("[ConfigureBuildSettings] 当前 Build Settings 场景列表:");
        foreach (var scene in EditorBuildSettings.scenes)
        {
            if (scene.enabled)
            {
                Debug.Log($"  - {scene.path} (enabled)");
            }
        }
    }
}
