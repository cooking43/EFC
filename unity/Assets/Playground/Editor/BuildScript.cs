using UnityEditor;
using UnityEngine;
using System.IO;

/// <summary>
/// P0 Headless 构建脚本
///
/// 用途：
/// 1. 从命令行构建 Linux headless 可执行文件
/// 2. 用于 ML-Agents 训练的无头模式构建
/// 3. 支持批量实验自动化
///
/// 使用方法（从项目根目录）：
///   bash scripts/build.sh
///
/// 或直接调用 Unity：
///   Unity -quit -batchmode -projectPath gewu/ \
///     -executeMethod BuildScript.Build \
///     -buildTarget StandaloneLinux64 \
///     -logFile build.log
/// </summary>
public class BuildScript
{
    /// <summary>
    /// 构建 Linux Headless 可执行文件
    /// </summary>
    [MenuItem("Build/Build Linux Headless")]
    public static void Build()
    {
        try
        {
            string buildPath = GetBuildPath();
            string executableName = "RL-Playground";

            Debug.Log($"[BuildScript] 开始构建...");
            Debug.Log($"[BuildScript] 目标路径: {buildPath}");
            Debug.Log($"[BuildScript] 可执行文件: {executableName}");

            // S2: 显式切换构建目标
            Debug.Log($"[BuildScript] 当前平台: {EditorUserBuildSettings.activeBuildTarget}");
            bool switched = EditorUserBuildSettings.SwitchActiveBuildTarget(
                BuildTargetGroup.Standalone,
                BuildTarget.StandaloneLinux64
            );
            Debug.Log($"[BuildScript] 切换到Linux64: {(switched ? "成功" : "已是目标平台")}");

            // S2: 强制使用Mono后端（避免IL2CPP依赖）
            PlayerSettings.SetScriptingBackend(
                BuildTargetGroup.Standalone,
                ScriptingImplementation.Mono2x
            );
            Debug.Log($"[BuildScript] 脚本后端: Mono2x");

            // 确保构建目录存在且可写
            string buildDir = Path.GetDirectoryName(buildPath);
            if (!Directory.Exists(buildDir))
            {
                Directory.CreateDirectory(buildDir);
                Debug.Log($"[BuildScript] 已创建构建目录: {buildDir}");
            }

            // 验证目录可写
            string testFile = Path.Combine(buildDir, ".buildtest");
            File.WriteAllText(testFile, "test");
            File.Delete(testFile);
            Debug.Log($"[BuildScript] 构建目录可写: ✓");

            // 获取所有启用的场景
            string[] scenes = GetEnabledScenes();
            if (scenes.Length == 0)
            {
                Debug.LogError("[BuildScript] ✗ 无可用场景！");
                EditorApplication.Exit(1);
                return;
            }
            Debug.Log($"[BuildScript] 包含场景数: {scenes.Length}");
            foreach (var scene in scenes)
            {
                Debug.Log($"[BuildScript]   - {scene}");
            }

            // T1: 标准player构建（用-nographics参数实现headless，不需要EnableHeadlessMode）
            BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions
            {
                scenes = scenes,
                locationPathName = Path.GetFullPath(buildPath),  // S2: 使用绝对路径
                target = BuildTarget.StandaloneLinux64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None  // Standard player, run with --no-graphics
            };

            Debug.Log($"[BuildScript] 构建选项:");
            Debug.Log($"[BuildScript]   - 目标: {buildPlayerOptions.target}");
            Debug.Log($"[BuildScript]   - 输出: {buildPlayerOptions.locationPathName}");
            Debug.Log($"[BuildScript]   - 选项: {buildPlayerOptions.options}");

            // 执行构建
            Debug.Log("[BuildScript] 执行构建中...");
            UnityEditor.Build.Reporting.BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
            UnityEditor.Build.Reporting.BuildSummary summary = report.summary;

            // S2: 详细输出构建结果
            Debug.Log($"[BuildScript] 构建结果: {summary.result}");
            Debug.Log($"[BuildScript]   - 输出路径: {summary.outputPath}");
            Debug.Log($"[BuildScript]   - 总大小: {summary.totalSize / (1024 * 1024):F2} MB");
            Debug.Log($"[BuildScript]   - 耗时: {summary.totalTime.TotalSeconds:F1} 秒");
            Debug.Log($"[BuildScript]   - 错误数: {summary.totalErrors}");
            Debug.Log($"[BuildScript]   - 警告数: {summary.totalWarnings}");

            // 输出所有构建步骤
            if (report.steps != null && report.steps.Length > 0)
            {
                Debug.Log($"[BuildScript] 构建步骤 ({report.steps.Length}):");
                foreach (var step in report.steps)
                {
                    Debug.Log($"[BuildScript]   - {step.name}: {step.duration.TotalSeconds:F2}s");
                }
            }

            // 检查构建结果
            if (summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded)
            {
                Debug.Log($"[BuildScript] ✓ 构建成功");

                // 验证输出文件存在
                if (File.Exists(summary.outputPath))
                {
                    Debug.Log($"[BuildScript] ✓ 可执行文件已生成: {summary.outputPath}");
                }
                else
                {
                    Debug.LogError($"[BuildScript] ✗ 可执行文件不存在: {summary.outputPath}");
                    EditorApplication.Exit(1);
                    return;
                }

                EditorApplication.Exit(0);
            }
            else
            {
                Debug.LogError($"[BuildScript] ✗ 构建失败: {summary.result}");

                // 输出所有错误消息
                if (report.summary.totalErrors > 0)
                {
                    Debug.LogError("[BuildScript] 错误详情:");
                    // BuildReport不直接提供错误消息，但会输出到控制台
                }

                EditorApplication.Exit(1);
            }
        }
        catch (System.Exception e)
        {
            // S2: 捕获并打印所有异常
            Debug.LogError($"[BuildScript] ✗ 构建异常: {e.GetType().Name}");
            Debug.LogError($"[BuildScript] 消息: {e.Message}");
            Debug.LogError($"[BuildScript] 堆栈:\n{e.StackTrace}");

            if (e.InnerException != null)
            {
                Debug.LogError($"[BuildScript] 内部异常: {e.InnerException.Message}");
            }

            EditorApplication.Exit(1);
        }
    }

    /// <summary>
    /// 获取构建输出路径（可通过命令行参数覆盖）
    /// </summary>
    private static string GetBuildPath()
    {
        // 默认路径
        string defaultPath = Path.Combine(
            Directory.GetParent(Application.dataPath).FullName,
            "Build",
            "Linux",
            "RL-Playground.x86_64"
        );

        // 从命令行参数读取（如果有）
        string[] args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "-buildPath" && i + 1 < args.Length)
            {
                string customPath = args[i + 1];
                Debug.Log($"[BuildScript] 使用自定义构建路径: {customPath}");
                return customPath;
            }
        }

        return defaultPath;
    }

    /// <summary>
    /// 获取构建设置中启用的所有场景
    /// R1 修复：强制使用 ShipDeck.unity（干净的训练场景，无 ReflectionProbe）
    /// </summary>
    private static string[] GetEnabledScenes()
    {
        // R1: 强制使用 ShipDeck.unity - 包含所有训练设施但无演示场景的 ReflectionProbe
        string shipDeckScene = "Assets/Playground/ShipDeck.unity";

        if (!File.Exists(shipDeckScene))
        {
            Debug.LogError($"[BuildScript] ShipDeck.unity 不存在: {shipDeckScene}");
            Debug.LogError("[BuildScript] 请确保场景文件存在");
            return new string[0];
        }

        Debug.Log($"[BuildScript] 使用训练场景: {shipDeckScene}");
        return new string[] { shipDeckScene };
    }

    /// <summary>
    /// 验证构建环境
    /// </summary>
    [MenuItem("Build/Validate Build Environment")]
    public static void ValidateBuildEnvironment()
    {
        Debug.Log("=== 构建环境验证 ===");

        // 检查场景
        string[] scenes = GetEnabledScenes();
        Debug.Log($"✓ 启用的场景数: {scenes.Length}");
        foreach (var scene in scenes)
        {
            Debug.Log($"  - {scene}");
        }

        // 检查 ML-Agents 包
        bool hasMLAgents = false;
        var packages = UnityEditor.PackageManager.PackageInfo.GetAllRegisteredPackages();
        foreach (var package in packages)
        {
            if (package.name.Contains("ml-agents"))
            {
                Debug.Log($"✓ ML-Agents: {package.displayName} ({package.version})");
                hasMLAgents = true;
            }
        }

        if (!hasMLAgents)
        {
            Debug.LogWarning("⚠ 未检测到 ML-Agents 包");
        }

        // 检查平台
        Debug.Log($"✓ 当前平台: {EditorUserBuildSettings.activeBuildTarget}");
        Debug.Log($"✓ Unity 版本: {Application.unityVersion}");

        Debug.Log("=== 验证完成 ===");
    }
}
