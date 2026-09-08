using UnityEngine;

/// <summary>
/// D2: 基于预测的前馈控制器（修复后版本）
/// 通过调整支撑腿膝盖高度来补偿Roll方向的倾斜
///
/// 物理原理（G1 实测验证）：
/// G1 测试结论：Unity 左手系，+roll（绕 Z 轴正向旋转）→ 右侧（+X）上升，左侧（-X）下沉
///
/// - 甲板 +roll（右侧高，左侧低）时，为使质心保持在支撑区内：
///   - 右腿支撑（左腿摆动）：右腿在上坡侧，需伸展膝盖（正补偿）推动质心向右
///   - 左腿支撑（右腿摆动）：左腿在下坡侧，需弯曲膝盖（负补偿）推动质心向右
///
/// 门控逻辑（D2 修复）：
/// - uf1 > threshold：左腿摆动 → 右腿支撑 → 补偿施加于右膝
/// - uf2 > threshold：右腿摆动 → 左腿支撑 → 补偿施加于左膝
///
/// D1: 关节索引通过 RobotRLAgent.IdxLeftKnee/IdxRightKnee 获取，不再硬编码
/// K3: 符号经 G1/G2a/G2b 测试验证正确（待闭环 A/B 测试最终确认）
/// </summary>
public class PredictiveFeedforwardController : MonoBehaviour
{
    private RobotRLAgent agent;  // D1: 引用 agent 获取关节索引


    [Header("补偿增益")]
    [Tooltip("Roll补偿增益（预测倾斜5度 → 补偿0.5度）")]
    public float K_roll = 0.1f;

    [Header("限制参数")]
    [Tooltip("最大补偿角度（度）")]
    public float max_roll_comp = 8.0f;

    [Tooltip("支撑判断阈值（uf > threshold 认为该腿正在摆动）")]
    public float support_threshold = 0.7f;

    [Header("滤波参数")]
    [Tooltip("滤波系数（0~1，越小越平滑但响应越慢）")]
    public float filter_alpha = 0.3f;

    // 滤波后的Roll误差
    private float filtered_roll_error = 0f;

    // D1: 初始化时获取 agent 引用
    void Start()
    {
        Initialize();
    }

    /// <summary>
    /// K1: 公开初始化方法供测试使用
    /// </summary>
    public void Initialize()
    {
        if (agent == null)
        {
            agent = GetComponent<RobotRLAgent>();
            if (agent == null)
            {
                Debug.LogError("[PredictiveFeedforwardController] 未找到 RobotRLAgent 组件！");
            }
        }
    }

    /// <summary>
    /// D1/D2: 计算基于预测的膝盖补偿控制量（修复后版本）
    /// </summary>
    /// <param name="predictedEuler">预测的欧拉角（度）</param>
    /// <param name="uf1">左腿摆动相位函数（0~1）</param>
    /// <param name="uf2">右腿摆动相位函数（0~1）</param>
    /// <param name="ActionNum">关节数量（12）</param>
    /// <returns>补偿控制量数组</returns>
    public float[] ComputeKneeCompensation(
        Vector3 predictedEuler,
        float uf1,
        float uf2,
        int ActionNum)
    {
        float[] compensation = new float[ActionNum];

        // D1: 验证关节索引已初始化
        if (agent == null || agent.IdxLeftKnee < 0 || agent.IdxRightKnee < 0)
        {
            Debug.LogWarning("[PredictiveFeedforwardController] 关节索引未初始化，跳过补偿计算");
            return compensation;  // 返回全零数组
        }

        // 提取预测的Roll误差（z轴）
        float raw_roll_error = predictedEuler.z;

        // 指数滤波（减少噪声）
        filtered_roll_error = filter_alpha * raw_roll_error + (1f - filter_alpha) * filtered_roll_error;

        // 计算基础补偿量（带饱和限制）
        float base_comp = K_roll * filtered_roll_error;  // K3: 符号已验证正确（待闭环确认）
        base_comp = Mathf.Clamp(base_comp, -max_roll_comp, max_roll_comp);

        // D2: 判断摆动腿（uf > threshold 表示正在摆动）
        bool leftSwing = uf1 > support_threshold;   // uf1 高 → 左腿摆动 → 右腿支撑
        bool rightSwing = uf2 > support_threshold;  // uf2 高 → 右腿摆动 → 左腿支撑

        // D2: 施加补偿到支撑腿的膝盖（修复后的物理逻辑）
        // 物理：roll > 0（向右倾，右侧低）时
        //   - 右腿支撑（在下坡侧）：应伸展（正补偿）推高右侧
        //   - 左腿支撑（在上坡侧）：应弯曲（负补偿）降低左侧

        if (leftSwing)  // 左腿摆动 → 右腿支撑
        {
            // D1: 使用按名称映射的索引，而非硬编码的 9
            // D2: 符号逻辑 - roll > 0 时 base_comp > 0，右腿应伸展（正补偿）
            compensation[agent.IdxRightKnee] = base_comp;
        }

        if (rightSwing)  // 右腿摆动 → 左腿支撑
        {
            // D1: 使用按名称映射的索引，而非硬编码的 3
            // D2: 符号逻辑 - roll > 0 时 base_comp > 0，左腿应弯曲（负补偿）
            compensation[agent.IdxLeftKnee] = -base_comp;
        }

        return compensation;
    }

    /// <summary>
    /// 重置滤波器（在Episode开始时调用）
    /// </summary>
    public void ResetFilter()
    {
        filtered_roll_error = 0f;
    }

    /// <summary>
    /// 获取当前的滤波后Roll误差（用于调试）
    /// </summary>
    public float GetFilteredRollError()
    {
        return filtered_roll_error;
    }

    /// <summary>
    /// 在Scene视图中可视化补偿信息
    /// </summary>
    void OnDrawGizmos()
    {
        if (!Application.isPlaying)
            return;

        // 绘制Roll误差指示
        Vector3 pos = transform.position + Vector3.up * 1.5f;

        // 颜色表示倾斜方向
        if (filtered_roll_error > 0)
            Gizmos.color = Color.red;    // 向右倾
        else
            Gizmos.color = Color.blue;   // 向左倾

        // 箭头长度表示误差大小
        float arrowLength = Mathf.Abs(filtered_roll_error) * 0.1f;
        Vector3 direction = filtered_roll_error > 0 ? Vector3.right : Vector3.left;
        Gizmos.DrawRay(pos, direction * arrowLength);
    }
}
