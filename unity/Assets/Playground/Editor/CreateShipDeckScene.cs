using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine.SceneManagement;

/// <summary>
/// L1: 程序化生成 ShipDeck.unity 独立实验场景
/// </summary>
public static class CreateShipDeckScene
{
    [MenuItem("Tools/Create ShipDeck Scene")]
    public static void CreateScene()
    {
        Debug.Log("[CreateShipDeckScene] 开始创建 ShipDeck.unity...");

        // 1. 先创建新的空场景
        Scene shipDeckScene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        Debug.Log("[CreateShipDeckScene] ✓ 创建空场景");

        // 2. Additive 打开 Playground.unity 读取 G1 机器人
        string playgroundPath = "Assets/Playground/Playground.unity";
        Scene playgroundScene = EditorSceneManager.OpenScene(playgroundPath, OpenSceneMode.Additive);

        if (!playgroundScene.IsValid())
        {
            Debug.LogError($"[CreateShipDeckScene] 无法打开场景: {playgroundPath}");
            return;
        }

        Debug.Log($"[CreateShipDeckScene] ✓ 打开 Playground.unity (Additive)");

        // 2. 使用反射查找 RobotRLAgent 类型
        System.Type agentType = null;
        foreach (var assembly in System.AppDomain.CurrentDomain.GetAssemblies())
        {
            agentType = assembly.GetType("RobotRLAgent");
            if (agentType != null) break;
        }

        if (agentType == null)
        {
            Debug.LogError("[CreateShipDeckScene] 未找到 RobotRLAgent 类型");
            EditorSceneManager.CloseScene(playgroundScene, true);
            return;
        }

        // 3. 查找 Robot==biped && ActionNum==12 的机器人，优先 G1
        GameObject g1Robot = null;
        System.Collections.Generic.List<GameObject> bipedRobots = new System.Collections.Generic.List<GameObject>();
        GameObject[] rootObjects = playgroundScene.GetRootGameObjects();

        foreach (GameObject rootObj in rootObjects)
        {
            if (rootObj.name.Contains("Camera") || rootObj.name.Contains("Light") ||
                rootObj.name == "Ground" || rootObj.name == "Canvas")
            {
                continue;
            }

            Component agent = rootObj.GetComponentInChildren(agentType);
            if (agent != null)
            {
                var robotField = agentType.GetField("Robot");
                var actionNumField = agentType.GetField("ActionNum");

                int robotValue = (int)robotField.GetValue(agent);
                int actionNum = (int)actionNumField.GetValue(agent);

                // biped(0) && ActionNum==12
                if (robotValue == 0 && actionNum == 12)
                {
                    bipedRobots.Add(rootObj);
                    Debug.Log($"[CreateShipDeckScene] 找到 biped 机器人: {rootObj.name} (Robot={robotValue}, ActionNum={actionNum})");
                }
            }
        }

        if (bipedRobots.Count == 0)
        {
            Debug.LogError("[CreateShipDeckScene] 未找到 biped(0) && ActionNum==12 的机器人");
            EditorSceneManager.CloseScene(playgroundScene, true);
            return;
        }

        // 优先选择名为 "G1" 的
        foreach (GameObject robot in bipedRobots)
        {
            if (robot.name == "G1")
            {
                g1Robot = robot;
                break;
            }
        }

        // 如果没有 G1，选第一个
        if (g1Robot == null)
        {
            g1Robot = bipedRobots[0];
        }

        Debug.Log($"[CreateShipDeckScene] ✓ 选中机器人: {g1Robot.name}");

        // 4. 复制 G1 机器人
        GameObject g1Copy = Object.Instantiate(g1Robot);
        g1Copy.name = "G1";
        Debug.Log($"[CreateShipDeckScene] ✓ 复制 G1 机器人");

        // 5. 将 G1 移动到新场景
        SceneManager.MoveGameObjectToScene(g1Copy, shipDeckScene);
        Debug.Log("[CreateShipDeckScene] ✓ G1 移至新场景");

        // 6. 关闭 Playground.unity
        EditorSceneManager.CloseScene(playgroundScene, false);
        Debug.Log("[CreateShipDeckScene] ✓ 关闭 Playground.unity");

        // AA1: 配置多个 TrainingArea（默认14个，沿X轴间隔20m）
        int numAreas = 14;
        float areaSpacing = 20f;  // Ground 12m宽，间隔20m留8m缓冲
        Debug.Log($"[CreateShipDeckScene] AA1: 准备生成 {numAreas} 个 TrainingArea，间隔 {areaSpacing}m");

        // AA1: 循环生成多个 TrainingArea
        for (int areaIndex = 0; areaIndex < numAreas; areaIndex++)
        {
            // AA1: 计算X轴偏移（居中）
            float xOffset = (areaIndex - (numAreas - 1) / 2f) * areaSpacing;

            // AA1: 为每个area复制一份G1
            GameObject g1ForThisArea;
            if (areaIndex == 0)
            {
                g1ForThisArea = g1Copy;  // 第一个area使用原始复制
            }
            else
            {
                g1ForThisArea = Object.Instantiate(g1Copy);
                g1ForThisArea.name = "G1";
                SceneManager.MoveGameObjectToScene(g1ForThisArea, shipDeckScene);
            }

            // 7. 创建 TrainingArea 根节点
            GameObject trainingArea = new GameObject($"TrainingArea_{areaIndex}");
            trainingArea.AddComponent<TrainingArea>();
            trainingArea.transform.position = new Vector3(xOffset, 0f, 0f);

            Debug.Log($"[CreateShipDeckScene] AA1[{areaIndex}]: 创建 TrainingArea，X偏移 = {xOffset:F1}m");

            // 8. 移动 G1 到 TrainingArea 下，并设置初始高度
            g1ForThisArea.transform.SetParent(trainingArea.transform, true);
            g1ForThisArea.transform.localPosition = new Vector3(0f, 1.0f, 0f);

            // ExperimentConfig must live beside RobotRLAgent: both the agent and
            // configuration code resolve collaborating components on this object.
            if (g1ForThisArea.GetComponent<ExperimentConfig>() == null)
            {
                g1ForThisArea.AddComponent<ExperimentConfig>();
            }

            Debug.Log($"[CreateShipDeckScene] AA1[{areaIndex}]: G1 移至 TrainingArea，初始高度 Y=1.0m");

            // L3: 配置 VectorObservationSize（根据观测开关动态计算）
            Component agentComp = g1ForThisArea.GetComponentInChildren(agentType);
            if (agentComp != null)
            {
                var actionNumField = agentType.GetField("ActionNum");
                var enableCoMField = agentType.GetField("enableCoMObservation");
                var enablePredField = agentType.GetField("enablePredictionObservation");
                var trainField = agentType.GetField("train");

                int actionNum = (int)actionNumField.GetValue(agentComp);
                bool enableCoM = (bool)enableCoMField.GetValue(agentComp);
                bool enablePred = (bool)enablePredField.GetValue(agentComp);

                // X1(a): 设置 train = true
                if (trainField != null)
                {
                    trainField.SetValue(agentComp, true);
                }

                int baseDim = 9 + 2 * actionNum;
                int comDim = enableCoM ? 5 : 0;
                int predDim = enablePred ? 24 : 0;
                int expectedDim = baseDim + comDim + predDim;

                // 设置 BehaviorParameters
                Component[] components = g1ForThisArea.GetComponents<Component>();
                foreach (Component comp in components)
                {
                    if (comp.GetType().Name == "BehaviorParameters")
                    {
                        var behaviorParamsType = comp.GetType();
                        var brainParamsProp = behaviorParamsType.GetProperty("BrainParameters");
                        object brainParams = brainParamsProp.GetValue(comp);
                        var brainParamsType = brainParams.GetType();
                        var vectorObsSizeField = brainParamsType.GetField("VectorObservationSize");
                        vectorObsSizeField.SetValue(brainParams, expectedDim);
                        brainParamsProp.SetValue(comp, brainParams);

                        var modelField = behaviorParamsType.GetField("m_Model",
                            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        if (modelField != null)
                        {
                            modelField.SetValue(comp, null);
                        }

                        var behaviorTypeField = behaviorParamsType.GetField("m_BehaviorType",
                            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                        if (behaviorTypeField != null)
                        {
                            behaviorTypeField.SetValue(comp, 0);  // Default
                        }

                        UnityEditor.EditorUtility.SetDirty(comp);
                        break;
                    }
                }
            }

            // 9. 创建 Ground (12m × 30m)
            GameObject ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.SetParent(trainingArea.transform, false);
            ground.transform.localPosition = Vector3.zero;
            ground.transform.localRotation = Quaternion.identity;
            ground.transform.localScale = new Vector3(1.2f, 1f, 3.0f);

            var meshCollider = ground.GetComponent<MeshCollider>();
            if (meshCollider != null) Object.DestroyImmediate(meshCollider);

            var boxCollider = ground.AddComponent<BoxCollider>();
            boxCollider.size = new Vector3(10f, 0.1f, 10f);
            boxCollider.center = new Vector3(0f, 0f, 0f);

            // 10. 添加 WaveSpectrumPlatformController
            var waveController = ground.AddComponent<WaveSpectrumPlatformController>();
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

            // 11. 添加 kinematic Rigidbody
            var rb = ground.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;

            Debug.Log($"[CreateShipDeckScene] AA1[{areaIndex}]: Ground + WaveController + Rigidbody 创建完成");
        }  // AA1: 循环结束

        Debug.Log($"[CreateShipDeckScene] AA1: ✓✓✓ 全部 {numAreas} 个 TrainingArea 创建完成");

        // 12. 创建 Directional Light
        GameObject light = new GameObject("Directional Light");
        Light lightComp = light.AddComponent<Light>();
        lightComp.type = LightType.Directional;
        lightComp.color = Color.white;
        lightComp.intensity = 1f;
        light.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        Debug.Log("[CreateShipDeckScene] ✓ 创建 Directional Light");

        // 13. 创建 Main Camera (不带 CameraFollow)
        GameObject camera = new GameObject("Main Camera");
        camera.tag = "MainCamera";
        Camera cameraComp = camera.AddComponent<Camera>();
        cameraComp.clearFlags = CameraClearFlags.Skybox;
        cameraComp.fieldOfView = 60f;
        camera.transform.position = new Vector3(0f, 5f, -10f);
        camera.transform.rotation = Quaternion.Euler(15f, 0f, 0f);
        Debug.Log("[CreateShipDeckScene] ✓ 创建 Main Camera (无 CameraFollow)");

        // 14. 保存场景
        string shipDeckPath = "Assets/Playground/ShipDeck.unity";
        EditorSceneManager.SaveScene(shipDeckScene, shipDeckPath);
        EditorSceneManager.OpenScene(shipDeckPath, OpenSceneMode.Single);

        Debug.Log($"[CreateShipDeckScene] ✓✓✓ 场景创建完成: {shipDeckPath}");
        Debug.Log($"[CreateShipDeckScene] AA1 场景结构: {numAreas} 个 TrainingArea");
        Debug.Log($"  - TrainingArea_0 ... TrainingArea_{numAreas-1} (沿X轴间隔{areaSpacing}m)");
        Debug.Log("    |- G1 (RobotRLAgent)");
        Debug.Log("    |- Ground (WaveSpectrumPlatformController, 12m×30m)");
        Debug.Log("    |- ExperimentConfig");
        Debug.Log("  - Directional Light");
        Debug.Log("  - Main Camera");

        // AA1: 场景自检（验证所有TrainingArea）
        Debug.Log("[CreateShipDeckScene] === AA1 场景自检 ===");

        TrainingArea[] trainingAreas = Object.FindObjectsOfType<TrainingArea>();
        if (trainingAreas.Length != numAreas)
        {
            Debug.LogError($"[CreateShipDeckScene] ✗ AA1自检失败：预期 {numAreas} 个 TrainingArea，实际找到 {trainingAreas.Length} 个");
            throw new System.Exception("AA1自检失败：TrainingArea数量不匹配");
        }

        Debug.Log($"[CreateShipDeckScene] ✓ AA1(1): 找到 {trainingAreas.Length} 个 TrainingArea");

        // AA1: 验证每个area的组件
        int validAreas = 0;
        foreach (var area in trainingAreas)
        {
            var ground = area.GetComponentInChildren<WaveSpectrumPlatformController>();
            var agent = area.GetComponentInChildren(agentType);
            var config = area.GetComponentInChildren<ExperimentConfig>();
            var collider = area.GetComponentInChildren<Collider>();

            if (ground != null && agent != null && config != null && collider != null)
            {
                validAreas++;
            }
            else
            {
                Debug.LogWarning($"[CreateShipDeckScene] ⚠ {area.name} 缺少组件: " +
                    $"Ground={ground != null}, Agent={agent != null}, Config={config != null}, Collider={collider != null}");
            }
        }

        if (validAreas != numAreas)
        {
            Debug.LogError($"[CreateShipDeckScene] ✗ AA1自检失败：仅 {validAreas}/{numAreas} 个 area 完整");
            throw new System.Exception("AA1自检失败：部分area缺少组件");
        }

        Debug.Log($"[CreateShipDeckScene] ✓ AA1(2): 全部 {numAreas} 个 area 组件完整");
        Debug.Log("[CreateShipDeckScene] ✓✓✓ AA1 场景自检通过");
    }
}
