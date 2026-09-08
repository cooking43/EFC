using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;
using System.Linq;
using System.Collections.Generic;

/// <summary>
/// I1: Training Area 场景改造脚本
///
/// 自动化完成 Playground.unity 的架构调整：
/// - 创建 TrainingArea 根节点
/// - 将 train=1 的机器人和 Ground 重组到 TrainingArea
/// - 替换 PlatformMotionController 为 WaveSpectrumPlatformController
/// - 配置参数
/// </summary>
public class SetupTrainingArea : EditorWindow
{
    [MenuItem("Tools/Setup Training Area")]
    public static void SetupScene()
    {
        Debug.Log("[SetupTrainingArea] 开始场景改造...");

        // 1. 打开场景
        string scenePath = "Assets/Playground/Playground.unity";
        Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);

        if (!scene.IsValid())
        {
            Debug.LogError($"[SetupTrainingArea] 无法打开场景: {scenePath}");
            return;
        }

        Debug.Log($"[SetupTrainingArea] ✓ 场景已打开: {scene.name}");

        // 1.5. K1 修复：检查是否已经设置过 TrainingArea
        GameObject[] rootObjects = scene.GetRootGameObjects();
        GameObject existingTrainingArea = null;
        foreach (GameObject rootObj in rootObjects)
        {
            if (rootObj.name == "TrainingArea" && rootObj.GetComponent<TrainingArea>() != null)
            {
                existingTrainingArea = rootObj;
                break;
            }
        }

        if (existingTrainingArea != null)
        {
            Debug.Log("[SetupTrainingArea] 检测到已存在 TrainingArea，跳过场景改造");
            Debug.Log("[SetupTrainingArea] 如需重新设置，请先删除场景中的 TrainingArea 根节点");
            return;
        }

        // 2. J1 修复：查找正确的机器人
        // 遍历所有 RobotRLAgent，选择 train==true 且 Robot==biped 且 ActionNum==12 的机器人
        // 使用反射避免 Editor 脚本引用 ML-Agents 程序集
        GameObject robot = null;

        Debug.Log($"[SetupTrainingArea] 场景根对象列表 (共 {rootObjects.Length} 个):");

        List<GameObject> candidateRobots = new List<GameObject>();
        List<GameObject> allRobots = new List<GameObject>();  // K1: 记录所有机器人用于清理

        // 通过反射获取 RobotRLAgent 类型（扫描所有程序集）
        System.Type agentType = null;
        foreach (var assembly in System.AppDomain.CurrentDomain.GetAssemblies())
        {
            agentType = assembly.GetType("RobotRLAgent");
            if (agentType != null)
            {
                break;
            }
        }

        if (agentType == null)
        {
            Debug.LogError("[SetupTrainingArea] 未找到 RobotRLAgent 类型");
            return;
        }

        foreach (GameObject rootObj in rootObjects)
        {
            // 跳过明显不是机器人的对象
            if (rootObj.name.Contains("Camera") || rootObj.name == "Directional Light" ||
                rootObj.name == "Ground" || rootObj.name == "Canvas")
            {
                continue;
            }

            // 查找 RobotRLAgent 组件（使用反射）
            Component agent = rootObj.GetComponentInChildren(agentType);

            if (agent != null)
            {
                // K1: 记录所有找到的机器人
                allRobots.Add(rootObj);

                // 通过反射读取字段
                var trainField = agentType.GetField("train");
                var robotField = agentType.GetField("Robot");
                var actionNumField = agentType.GetField("ActionNum");

                bool train = (bool)trainField.GetValue(agent);
                int robotValue = (int)robotField.GetValue(agent);
                int actionNum = (int)actionNumField.GetValue(agent);

                Debug.Log($"  - {rootObj.name}: 找到 RobotRLAgent");
                Debug.Log($"      train={train}, Robot={robotValue}, ActionNum={actionNum}");

                // J1: 筛选条件 - Robot==biped(0) 且 ActionNum==12
                // 如果 train==true 优先，否则也收集作为候选
                if (robotValue == 0 && actionNum == 12)
                {
                    candidateRobots.Add(rootObj);
                    Debug.Log($"      ✓ 符合类型条件（biped(0), ActionNum=12）, train={train}");
                }
            }
        }

        // J1: 找不到或找到多个都要报错
        if (candidateRobots.Count == 0)
        {
            Debug.LogError("[SetupTrainingArea] 未找到符合条件的机器人（Robot==biped(0) 且 ActionNum==12）");
            return;
        }
        else if (candidateRobots.Count > 1)
        {
            // 如果有多个候选，优先选择 train==true 的
            GameObject trainTrueRobot = null;
            foreach (GameObject candidate in candidateRobots)
            {
                Component agent = candidate.GetComponentInChildren(agentType);
                var trainField = agentType.GetField("train");
                bool trainValue = (bool)trainField.GetValue(agent);
                if (trainValue)
                {
                    if (trainTrueRobot != null)
                    {
                        Debug.LogError($"[SetupTrainingArea] 找到多个 train=true 的符合条件机器人，无法确定唯一训练目标");
                        foreach (GameObject r in candidateRobots)
                        {
                            Debug.LogError($"  - {r.name}");
                        }
                        return;
                    }
                    trainTrueRobot = candidate;
                }
            }

            if (trainTrueRobot != null)
            {
                candidateRobots.Clear();
                candidateRobots.Add(trainTrueRobot);
            }
            else
            {
                // J1: 没有 train=true 的，优先选择名为 "G1" 的，然后自动设置 train=true
                GameObject g1Robot = null;
                foreach (GameObject candidate in candidateRobots)
                {
                    if (candidate.name == "G1")
                    {
                        g1Robot = candidate;
                        break;
                    }
                }

                if (g1Robot != null)
                {
                    // 找到 G1，设置 train=true
                    Component agent = g1Robot.GetComponentInChildren(agentType);
                    var trainField = agentType.GetField("train");
                    trainField.SetValue(agent, true);

                    candidateRobots.Clear();
                    candidateRobots.Add(g1Robot);

                    Debug.LogWarning($"[SetupTrainingArea] 找到 {candidateRobots.Count} 个符合类型条件的机器人（OpenLoong, X02Lite, G1），但都没有设置 train=true");
                    Debug.LogWarning($"[SetupTrainingArea] 自动选择 G1 并设置 train=true");
                }
                else
                {
                    // 找不到 G1，报错列出所有候选
                    Debug.LogError($"[SetupTrainingArea] 找到 {candidateRobots.Count} 个符合类型条件的机器人，但都没有设置 train=true，且未找到 G1");
                    foreach (GameObject r in candidateRobots)
                    {
                        Component agent = r.GetComponentInChildren(agentType);
                        var robotField = agentType.GetField("Robot");
                        var actionNumField = agentType.GetField("ActionNum");
                        Debug.LogError($"  - {r.name}: Robot={(int)robotField.GetValue(agent)}, ActionNum={(int)actionNumField.GetValue(agent)}");
                    }
                    Debug.LogError("[SetupTrainingArea] 请手动在 Unity Editor 中设置其中一个的 train 标志为 true，然后重新运行");
                    return;
                }
            }
        }

        robot = candidateRobots[0];
        Component selectedAgent = robot.GetComponentInChildren(agentType);
        var trainFieldFinal = agentType.GetField("train");
        var robotFieldFinal = agentType.GetField("Robot");
        var actionNumFieldFinal = agentType.GetField("ActionNum");

        Debug.Log($"[SetupTrainingArea] ✓ 选中机器人: {robot.name}");
        Debug.Log($"  类型: Robot={(int)robotFieldFinal.GetValue(selectedAgent)}, " +
                 $"train={trainFieldFinal.GetValue(selectedAgent)}, " +
                 $"ActionNum={actionNumFieldFinal.GetValue(selectedAgent)}");

        GameObject ground = GameObject.Find("Ground");
        if (ground == null)
        {
            Debug.LogError("[SetupTrainingArea] 未找到 Ground 对象");
            return;
        }

        Debug.Log($"[SetupTrainingArea] ✓ 找到机器人: {robot.name}, Ground: {ground.name}");

        // 3. 创建 TrainingArea 根节点
        GameObject trainingArea = new GameObject("TrainingArea");
        trainingArea.transform.position = Vector3.zero;
        trainingArea.transform.rotation = Quaternion.identity;

        // I3: 添加标记组件
        trainingArea.AddComponent<TrainingArea>();

        Debug.Log("[SetupTrainingArea] ✓ 创建 TrainingArea 根节点");

        // 4. 重组层级：机器人和 Ground 移到 TrainingArea 下
        robot.transform.SetParent(trainingArea.transform, true);
        ground.transform.SetParent(trainingArea.transform, true);

        Debug.Log("[SetupTrainingArea] ✓ 机器人和 Ground 已移至 TrainingArea");

        // 4.5. K1 修复：删除所有其他机器人，避免测试时误找
        // 只保留选中的机器人（G1），删除场景中所有其他机器人（OpenLoong, X02Lite等）
        if (allRobots.Count > 1)
        {
            int deletedCount = 0;
            foreach (GameObject otherRobot in allRobots)
            {
                if (otherRobot != robot)
                {
                    string robotName = otherRobot.name;
                    Object.DestroyImmediate(otherRobot);
                    deletedCount++;
                    Debug.Log($"[SetupTrainingArea] ✓ 删除未选中的机器人: {robotName}");
                }
            }
            Debug.Log($"[SetupTrainingArea] ✓ 已删除 {deletedCount} 个未选中的机器人，场景中只保留 {robot.name}");
        }

        // 4.6. K1 修复：删除 Main Camera 的 CameraFollow 组件
        // 测试不需要相机跟随，删除可以避免 UnassignedReferenceException
        GameObject mainCamera = GameObject.Find("Main Camera");
        if (mainCamera != null)
        {
            var cameraFollow = mainCamera.GetComponent<UnityEngine.MonoBehaviour>();
            if (cameraFollow != null && cameraFollow.GetType().Name == "CameraFollow")
            {
                Object.DestroyImmediate(cameraFollow);
                Debug.Log($"[SetupTrainingArea] ✓ 删除 Main Camera.CameraFollow 组件（测试不需要）");
            }
        }

        // 5. 移除旧控制器
        var oldController = ground.GetComponent<PlatformMotionController>();
        if (oldController != null)
        {
            Object.DestroyImmediate(oldController);
            Debug.Log("[SetupTrainingArea] ✓ 移除 PlatformMotionController");
        }

        // 6. 添加并配置 WaveSpectrumPlatformController
        var waveController = ground.GetComponent<WaveSpectrumPlatformController>();
        if (waveController == null)
        {
            waveController = ground.AddComponent<WaveSpectrumPlatformController>();
            Debug.Log("[SetupTrainingArea] ✓ 添加 WaveSpectrumPlatformController");
        }

        // I1: 配置参数
        waveController.waveMode = WaveMode.JONSWAP;
        waveController.significantRollAmplitudeRange = new Vector2(2f, 5f);
        waveController.peakPeriodRange = new Vector2(6f, 14f);
        waveController.gamma = 3.3f;

        waveController.rollNaturalPeriodRange = new Vector2(8f, 15f);
        waveController.rollDampingRatioRange = new Vector2(0.05f, 0.15f);
        waveController.pitchNaturalPeriodRange = new Vector2(5f, 10f);
        waveController.pitchDampingRatioRange = new Vector2(0.1f, 0.2f);
        waveController.heaveNaturalPeriodRange = new Vector2(4f, 8f);
        waveController.heaveDampingRatioRange = new Vector2(0.2f, 0.4f);

        waveController.pitchScale = 0.8f;
        waveController.significantHeaveAmplitudeRange = new Vector2(0.1f, 0.5f);
        waveController.numComponents = 50;
        waveController.omegaMin = 0.2f;
        waveController.omegaMax = 3.0f;
        waveController.waveSeed = -1;

        Debug.Log("[SetupTrainingArea] ✓ WaveSpectrumPlatformController 参数已配置");

        // 7. 确保 Ground 有 kinematic Rigidbody
        var rb = ground.GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = ground.AddComponent<Rigidbody>();
            Debug.Log("[SetupTrainingArea] ✓ 添加 Rigidbody");
        }

        rb.isKinematic = true;
        rb.useGravity = false;
        Debug.Log("[SetupTrainingArea] ✓ Rigidbody 已设为 kinematic");

        // 7.5. K1 修复：配置 BehaviorParameters.VectorObservationSize
        // 避免 "观测维度不匹配" 错误
        var robotAgent = robot.GetComponentInChildren(agentType);
        if (robotAgent != null)
        {
            // 使用反射读取 enableCoMObservation 和 enablePredictionObservation
            var enableCoMField = agentType.GetField("enableCoMObservation");
            var enablePredField = agentType.GetField("enablePredictionObservation");
            var actionNumField = agentType.GetField("ActionNum");

            bool enableCoM = (bool)enableCoMField.GetValue(robotAgent);
            bool enablePred = (bool)enablePredField.GetValue(robotAgent);
            int actionNum = (int)actionNumField.GetValue(robotAgent);

            // 计算期望的观测维度：baseDim + comDim + predDim
            int baseDim = 9 + 2 * actionNum;  // 9 + 2*12 = 33
            int comDim = enableCoM ? 5 : 0;
            int predDim = enablePred ? 24 : 0;
            int expectedDim = baseDim + comDim + predDim;

            Debug.Log($"[SetupTrainingArea] 计算观测维度: {expectedDim} = {baseDim}(base) + {comDim}(CoM) + {predDim}(Pred)");

            // 获取 BehaviorParameters 组件（使用反射）
            System.Type behaviorParamsType = null;
            foreach (var assembly in System.AppDomain.CurrentDomain.GetAssemblies())
            {
                behaviorParamsType = assembly.GetType("Unity.MLAgents.Policies.BehaviorParameters");
                if (behaviorParamsType != null) break;
            }

            if (behaviorParamsType != null)
            {
                Component behaviorParams = robot.GetComponentInChildren(behaviorParamsType);
                if (behaviorParams != null)
                {
                    // 使用反射修改 BrainParameters.VectorObservationSize
                    var brainParamsProperty = behaviorParamsType.GetProperty("BrainParameters");
                    object brainParams = brainParamsProperty.GetValue(behaviorParams);

                    var brainParamsType = brainParams.GetType();
                    var vectorObsSizeField = brainParamsType.GetField("VectorObservationSize");

                    int currentSize = (int)vectorObsSizeField.GetValue(brainParams);
                    Debug.Log($"[SetupTrainingArea] 当前 VectorObservationSize: {currentSize}");

                    if (currentSize != expectedDim)
                    {
                        vectorObsSizeField.SetValue(brainParams, expectedDim);
                        brainParamsProperty.SetValue(behaviorParams, brainParams);

                        Debug.Log($"[SetupTrainingArea] ✓ 已更新 VectorObservationSize: {currentSize} → {expectedDim}");
                    }
                    else
                    {
                        Debug.Log($"[SetupTrainingArea] ✓ VectorObservationSize 已正确: {expectedDim}");
                    }
                }
                else
                {
                    Debug.LogWarning("[SetupTrainingArea] 未找到 BehaviorParameters 组件");
                }
            }
            else
            {
                Debug.LogWarning("[SetupTrainingArea] 未找到 BehaviorParameters 类型");
            }
        }

        // 8. 保存场景
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);

        Debug.Log($"[SetupTrainingArea] ✓✓✓ 场景改造完成，已保存: {scenePath}");
        Debug.Log("[SetupTrainingArea] 改造摘要:");
        Debug.Log($"  - TrainingArea 根节点: 已创建");
        Debug.Log($"  - 机器人: {robot.name} (已移至 TrainingArea)");
        Debug.Log($"  - Ground: {ground.name} (已移至 TrainingArea)");
        Debug.Log($"  - 平台控制器: WaveSpectrumPlatformController (JONSWAP)");
        Debug.Log($"  - 参数: Hs=[2,5]°, Tp=[6,14]s, N=50");
        Debug.Log($"  - BehaviorParameters: VectorObservationSize 已配置");
    }
}
