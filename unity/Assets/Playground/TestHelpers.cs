using UnityEngine;

/// <summary>
/// L3: 测试辅助工具（临时修复）
/// </summary>
public static class TestHelpers
{
    /// <summary>
    /// L3: 临时为 Ground 添加 BoxCollider（如果没有）
    /// 在场景手动重新生成前的临时修复
    /// </summary>
    public static void EnsureGroundCollider(GameObject ground)
    {
        if (ground == null) return;

        // 检查是否已有Collider
        Collider existingCollider = ground.GetComponentInChildren<Collider>();
        if (existingCollider != null)
        {
            Debug.Log($"[TestHelpers] Ground 已有 Collider: {existingCollider.GetType().Name}");
            return;
        }

        // 临时添加 BoxCollider
        BoxCollider boxCollider = ground.AddComponent<BoxCollider>();
        boxCollider.size = new Vector3(10f, 0.1f, 10f);
        boxCollider.center = Vector3.zero;

        Debug.LogWarning($"[TestHelpers] ⚠ 临时添加 BoxCollider 到 Ground（场景需手动重新生成！）");
        Debug.LogWarning($"[TestHelpers] 运行 Tools > Create ShipDeck Scene 重新生成场景");
    }
}
