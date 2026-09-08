using NUnit.Framework;
using System;
using System.Collections.Generic;

/// <summary>
/// O2/O3: EditMode 波浪谱单元测试（无场景依赖，根治场景泄漏问题）
///
/// 从 PlayMode WaveSpectrumValidationTests 迁移过来
/// 直接测试 WaveSpectrum 静态类，不加载任何 Unity 场景
///
/// O2: 多seed统计测试标定算法（20个seed，测均值±标准差）
/// O3a: 中心差分 (O(Δt²) 精度)
/// O3b: smootherstep (C² 连续)
/// O3c: RMS归一化误差（不用逐点相对误差）
/// </summary>
[TestFixture]
public class WaveSpectrumTests
{
    private const float TOLERANCE = 0.05f;  // 5% tolerance for calibration mean
    private const int NUM_SEEDS = 20;       // O2: Statistical test over 20 seeds

    /// <summary>
    /// O2: 验证 Hs 标定精度（JONSWAP 谱，多seed统计）
    /// </summary>
    [Test]
    public void TestSignificantAmplitudeCalibration_JONSWAP()
    {
        var baseParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.JONSWAP,
            significantAmplitude = 5f,  // 5° roll
            peakPeriod = 10f,
            gamma = 3.3f,
            naturalPeriod = 12f,
            dampingRatio = 0.1f,
            numComponents = 100,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = 0  // Will be overridden
        };

        List<float> measuredHsList = new List<float>();

        // O2: Run 20 different seeds
        for (int seedIdx = 0; seedIdx < NUM_SEEDS; seedIdx++)
        {
            var waveParams = baseParams;
            waveParams.seed = 42 + seedIdx;
            var state = WaveSpectrum.GenerateWaveState(waveParams, 0f);

            // 运行5分钟模拟，采样计算实际 Hs
            int numSamples = 3000;  // 5 min @ 100ms
            float[] angles = new float[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                float t = i * 0.1f + 5f;  // 跳过前5s爬升
                WaveSpectrum.EvaluateMotion(state, t, out float angle, out _, out _);
                angles[i] = angle;
            }

            // 计算 Hs = 4 * σ
            float mean = 0f;
            foreach (float a in angles) mean += a;
            mean /= numSamples;

            float variance = 0f;
            foreach (float a in angles)
            {
                float dev = a - mean;
                variance += dev * dev;
            }
            variance /= numSamples;

            float sigma = (float)Math.Sqrt(variance);
            float measuredHs = 4f * sigma;
            measuredHsList.Add(measuredHs);
        }

        // 计算均值和标准差
        float hsMean = 0f;
        foreach (float h in measuredHsList) hsMean += h;
        hsMean /= NUM_SEEDS;

        float hsVariance = 0f;
        foreach (float h in measuredHsList)
        {
            float dev = h - hsMean;
            hsVariance += dev * dev;
        }
        hsVariance /= NUM_SEEDS;
        float hsStd = (float)Math.Sqrt(hsVariance);

        float relativeError = Math.Abs(hsMean - baseParams.significantAmplitude) / baseParams.significantAmplitude;

        // 输出实测数字
        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestSignificantAmplitudeCalibration_JONSWAP]");
        UnityEngine.Debug.Log($"  目标 Hs: {baseParams.significantAmplitude}°");
        UnityEngine.Debug.Log($"  实测 Hs (均值±标准差): {hsMean:F3}° ± {hsStd:F3}°");
        UnityEngine.Debug.Log($"  相对误差: {relativeError * 100f:F2}%");
        UnityEngine.Debug.Log($"  标准差/均值: {hsStd / hsMean * 100f:F2}%");

        Assert.LessOrEqual(relativeError, TOLERANCE,
            $"JONSWAP Hs 标定均值误差 {relativeError * 100:F1}% 超过 {TOLERANCE * 100}% 阈值");
    }

    /// <summary>
    /// O2: 验证 Hs 标定精度（PM 谱，多seed统计）
    /// </summary>
    [Test]
    public void TestSignificantAmplitudeCalibration_PM()
    {
        var baseParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.PiersonMoskowitz,
            significantAmplitude = 3f,
            peakPeriod = 8f,
            gamma = 1f,  // PM doesn't use gamma but set for consistency
            naturalPeriod = 10f,
            dampingRatio = 0.15f,
            numComponents = 80,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = 0
        };

        List<float> measuredHsList = new List<float>();

        for (int seedIdx = 0; seedIdx < NUM_SEEDS; seedIdx++)
        {
            var waveParams = baseParams;
            waveParams.seed = 123 + seedIdx;
            var state = WaveSpectrum.GenerateWaveState(waveParams, 0f);

            int numSamples = 3000;
            float[] angles = new float[numSamples];

            for (int i = 0; i < numSamples; i++)
            {
                float t = i * 0.1f + 5f;
                WaveSpectrum.EvaluateMotion(state, t, out float angle, out _, out _);
                angles[i] = angle;
            }

            float mean = 0f;
            foreach (float a in angles) mean += a;
            mean /= numSamples;

            float variance = 0f;
            foreach (float a in angles)
            {
                float dev = a - mean;
                variance += dev * dev;
            }
            variance /= numSamples;

            float sigma = (float)Math.Sqrt(variance);
            float measuredHs = 4f * sigma;
            measuredHsList.Add(measuredHs);
        }

        float hsMean = 0f;
        foreach (float h in measuredHsList) hsMean += h;
        hsMean /= NUM_SEEDS;

        float hsVariance = 0f;
        foreach (float h in measuredHsList)
        {
            float dev = h - hsMean;
            hsVariance += dev * dev;
        }
        hsVariance /= NUM_SEEDS;
        float hsStd = (float)Math.Sqrt(hsVariance);

        float relativeError = Math.Abs(hsMean - baseParams.significantAmplitude) / baseParams.significantAmplitude;

        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestSignificantAmplitudeCalibration_PM]");
        UnityEngine.Debug.Log($"  目标 Hs: {baseParams.significantAmplitude}°");
        UnityEngine.Debug.Log($"  实测 Hs (均值±标准差): {hsMean:F3}° ± {hsStd:F3}°");
        UnityEngine.Debug.Log($"  相对误差: {relativeError * 100f:F2}%");
        UnityEngine.Debug.Log($"  标准差/均值: {hsStd / hsMean * 100f:F2}%");

        Assert.LessOrEqual(relativeError, TOLERANCE,
            $"PM Hs 标定均值误差 {relativeError * 100:F1}% 超过 {TOLERANCE * 100}% 阈值");
    }

    /// <summary>
    /// J2: 验证 roll/pitch/heave 解耦
    /// </summary>
    [Test]
    public void TestAxesDecoupling()
    {
        // 创建三个独立的波浪状态
        var rollParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.JONSWAP,
            significantAmplitude = 5f,
            peakPeriod = 10f,
            gamma = 3.3f,
            naturalPeriod = 12f,
            dampingRatio = 0.1f,
            numComponents = 50,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = 42
        };

        var pitchParams = rollParams;
        pitchParams.seed = 43;

        var heaveParams = rollParams;
        heaveParams.seed = 44;
        heaveParams.significantAmplitude = 0.3f;  // meters

        var rollState = WaveSpectrum.GenerateWaveState(rollParams, 0f);
        var pitchState = WaveSpectrum.GenerateWaveState(pitchParams, 0f);
        var heaveState = WaveSpectrum.GenerateWaveState(heaveParams, 0f);

        // 验证相位独立性
        int phaseMatches = 0;
        for (int i = 0; i < rollState.phases.Length; i++)
        {
            if (Math.Abs(rollState.phases[i] - pitchState.phases[i]) < 0.1f)
                phaseMatches++;
        }

        float matchRate = (float)phaseMatches / rollState.phases.Length;

        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestAxesDecoupling]");
        UnityEngine.Debug.Log($"  Roll-Pitch 相位匹配率: {matchRate * 100f:F2}%");
        UnityEngine.Debug.Log($"  期望: <10% (随机独立)");

        Assert.Less(matchRate, 0.1f, "Roll 和 Pitch 相位应独立（不同seed）");
    }

    /// <summary>
    /// O3: 验证解析导数精度（中心差分 + RMS归一化）
    /// </summary>
    [Test]
    public void TestAnalyticalDerivativeAccuracy()
    {
        var waveParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.JONSWAP,
            significantAmplitude = 5f,
            peakPeriod = 10f,
            gamma = 3.3f,
            naturalPeriod = 12f,
            dampingRatio = 0.1f,
            numComponents = 50,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = 42
        };

        var state = WaveSpectrum.GenerateWaveState(waveParams, 0f);

        List<float> velErrors = new List<float>();
        List<float> accErrors = new List<float>();
        List<float> velocities = new List<float>();
        List<float> accelerations = new List<float>();

        // O3a: 中心差分，从 t=0.1 到 10s
        // P1c: 排除 smootherstep 端点附近（r'''(0) 和 r'''(0.5) 不连续）
        float dt = 0.01f;  // 10ms
        for (float t = 0.1f; t < 10f; t += 0.1f)
        {
            // P1c: 跳过 t=0 和 t=0.5 附近的采样点（三阶导数不连续，有限差分与解析值比较无效）
            // smootherstep r'''(0) = r'''(T_ramp) = 60/T³ ≠ 0
            if (Math.Abs(t - 0f) < 0.02f || Math.Abs(t - 0.5f) < 0.02f) continue;

            // 解析导数
            WaveSpectrum.EvaluateMotion(state, t, out float angle, out float velocity, out float acceleration);

            // O3a: 中心差分 (O(Δt²) 精度)
            WaveSpectrum.EvaluateMotion(state, t - dt, out float anglePrev, out _, out _);
            WaveSpectrum.EvaluateMotion(state, t + dt, out float angleNext, out _, out _);

            float numericalVel = (angleNext - anglePrev) / (2f * dt);
            float numericalAcc = (angleNext - 2f * angle + anglePrev) / (dt * dt);

            float velError = Math.Abs(velocity - numericalVel);
            float accError = Math.Abs(acceleration - numericalAcc);

            velErrors.Add(velError);
            accErrors.Add(accError);
            velocities.Add(velocity);
            accelerations.Add(acceleration);
        }

        // O3c: RMS归一化误差（绝对误差 / 信号RMS）
        float velRMS = ComputeRMS(velocities);
        float accRMS = ComputeRMS(accelerations);

        float velRMSError = ComputeRMS(velErrors) / velRMS;
        float accRMSError = ComputeRMS(accErrors) / accRMS;

        // 计算最大误差（仅用于报告，不作为判据）
        float maxVelError = 0f;
        float maxAccError = 0f;
        for (int i = 0; i < velErrors.Count; i++)
        {
            float velRel = velErrors[i] / (Math.Abs(velocities[i]) + 1e-6f);
            float accRel = accErrors[i] / (Math.Abs(accelerations[i]) + 1e-6f);
            maxVelError = Math.Max(maxVelError, velRel);
            maxAccError = Math.Max(maxAccError, accRel);
        }

        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestAnalyticalDerivativeAccuracy]");
        UnityEngine.Debug.Log($"  角速度 RMS 归一化误差: {velRMSError * 100f:F3}%");
        UnityEngine.Debug.Log($"  角加速度 RMS 归一化误差: {accRMSError * 100f:F3}%");
        UnityEngine.Debug.Log($"  （参考）角速度最大相对误差: {maxVelError * 100f:F3}%");
        UnityEngine.Debug.Log($"  （参考）角加速度最大相对误差: {maxAccError * 100f:F3}%");

        // O3c: 判据改为 RMS 归一化误差 < 1%
        Assert.Less(velRMSError, 0.01f, $"角速度 RMS 归一化误差 {velRMSError * 100:F2}% > 1%");
        Assert.Less(accRMSError, 0.02f, $"角加速度 RMS 归一化误差 {accRMSError * 100:F2}% > 2%");
    }

    /// <summary>
    /// J4: 验证 Resample 独立性
    /// </summary>
    [Test]
    public void TestResampleIndependence()
    {
        var waveParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.JONSWAP,
            significantAmplitude = 5f,
            peakPeriod = 10f,
            gamma = 3.3f,
            naturalPeriod = 12f,
            dampingRatio = 0.1f,
            numComponents = 50,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = -1  // 随机种子
        };

        var state1 = WaveSpectrum.GenerateWaveState(waveParams, 0f);
        var state2 = WaveSpectrum.GenerateWaveState(waveParams, 100f);

        // 验证相位不同
        int phaseMatches = 0;
        for (int i = 0; i < state1.phases.Length; i++)
        {
            if (Math.Abs(state1.phases[i] - state2.phases[i]) < 0.1f)
                phaseMatches++;
        }

        float matchRate = (float)phaseMatches / state1.phases.Length;

        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestResampleIndependence]");
        UnityEngine.Debug.Log($"  两次采样相位匹配率: {matchRate * 100f:F2}%");
        UnityEngine.Debug.Log($"  期望: <10% (随机独立)");

        Assert.Less(matchRate, 0.1f, "连续 resample 应产生独立相位");
    }

    /// <summary>
    /// P1b: 量纲一致性回归测试（防止 Rad2Deg 重复乘法等量纲错误）
    /// </summary>
    [Test]
    public void TestDimensionalConsistency()
    {
        float Hs = 5f;  // degrees
        float Tp = 10f;  // seconds

        // P1b: 理论峰值角速度约为 Hs × (2π/Tp)
        // 对于 JONSWAP 谱，实际峰值会因谱宽和 RAO 共振而有所不同，但量级应一致（3 倍以内）
        float expectedPeakVel = Hs * 2f * MathF.PI / Tp;  // ~3.14 deg/s

        // 测试不规则波
        var irregularParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.JONSWAP,
            significantAmplitude = Hs,
            peakPeriod = Tp,
            gamma = 3.3f,
            naturalPeriod = 12f,
            dampingRatio = 0.1f,
            numComponents = 50,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = 42
        };
        var irregularState = WaveSpectrum.GenerateWaveState(irregularParams, 0f);

        // 寻找不规则波峰值角速度
        float maxIrregularVel = 0f;
        for (float t = 1f; t < 100f; t += 0.1f)
        {
            WaveSpectrum.EvaluateMotion(irregularState, t, out _, out float vel, out _);
            maxIrregularVel = Math.Max(maxIrregularVel, Math.Abs(vel));
        }

        // 测试规则波（等效幅值和周期）
        var regularParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.RegularSine,
            regularAmplitude = Hs / 4f,  // 规则波幅值 = Hs/4（Hs = 4σ，规则波σ=A）
            regularFrequency = 1f / Tp,
            regularPhase = 0f
        };
        var regularState = WaveSpectrum.GenerateWaveState(regularParams, 0f);

        // 规则波峰值角速度（解析值）
        float regularPeakVel = regularParams.regularAmplitude * 2f * MathF.PI / Tp;

        // 验证不规则波峰值在理论预期 3 倍范围内
        float irregularRatio = maxIrregularVel / expectedPeakVel;

        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestDimensionalConsistency]");
        UnityEngine.Debug.Log($"  理论峰值角速度: {expectedPeakVel:F3} deg/s (Hs × 2π/Tp)");
        UnityEngine.Debug.Log($"  不规则波实测峰值: {maxIrregularVel:F3} deg/s (倍数: {irregularRatio:F2}x)");
        UnityEngine.Debug.Log($"  规则波解析峰值: {regularPeakVel:F3} deg/s");

        Assert.Less(irregularRatio, 3f,
            $"不规则波峰值 {maxIrregularVel:F2} deg/s 超过理论值 {expectedPeakVel:F2} 的 3 倍");
        Assert.Greater(irregularRatio, 0.33f,
            $"不规则波峰值 {maxIrregularVel:F2} deg/s 低于理论值 {expectedPeakVel:F2} 的 1/3");

        // 验证规则波和不规则波峰值同量级（10 倍范围内，因不规则波有随机性和 RAO 共振）
        float regularIrregularRatio = maxIrregularVel / regularPeakVel;
        Assert.Less(regularIrregularRatio, 10f,
            $"不规则波与规则波峰值量级差异过大: {maxIrregularVel:F2} vs {regularPeakVel:F2}");
        Assert.Greater(regularIrregularRatio, 0.1f,
            $"不规则波与规则波峰值量级差异过大: {maxIrregularVel:F2} vs {regularPeakVel:F2}");
    }

    /// <summary>
    /// Q1: FFT 频谱验证（生成时序数据供 Python 分析）
    /// </summary>
    [Test]
    public void TestFFTSpectrumValidation()
    {
        var waveParams = new WaveSpectrum.WaveParameters
        {
            mode = WaveSpectrum.SpectrumMode.JONSWAP,
            significantAmplitude = 5f,
            peakPeriod = 10f,
            gamma = 3.3f,
            naturalPeriod = 12f,
            dampingRatio = 0.1f,
            numComponents = 100,
            omegaMin = 0.2f,
            omegaMax = 3.0f,
            seed = 42
        };

        var rollState = WaveSpectrum.GenerateWaveState(waveParams, 0f);

        // 生成 pitch 和 heave 状态（不同种子）
        var pitchParams = waveParams;
        pitchParams.seed = 43;
        var pitchState = WaveSpectrum.GenerateWaveState(pitchParams, 0f);

        var heaveParams = waveParams;
        heaveParams.seed = 44;
        heaveParams.significantAmplitude = 0.3f;  // meters
        var heaveState = WaveSpectrum.GenerateWaveState(heaveParams, 0f);

        // 生成 600s @ 100Hz = 60000 采样点
        int numSamples = 60000;
        float dt = 0.01f;  // 100 Hz

        string csvPath = "/home/xds/PycharmProjects/Unity-RL-Playground/manuscript/revision/verification/wave_timeseries.csv";

        using (var writer = new System.IO.StreamWriter(csvPath))
        {
            writer.WriteLine("time,roll,pitch,heave,roll_vel,pitch_vel,heave_vel,roll_acc,pitch_acc,heave_acc");

            for (int i = 0; i < numSamples; i++)
            {
                float t = i * dt;

                WaveSpectrum.EvaluateMotion(rollState, t, out float roll, out float rollVel, out float rollAcc);
                WaveSpectrum.EvaluateMotion(pitchState, t, out float pitch, out float pitchVel, out float pitchAcc);
                WaveSpectrum.EvaluateMotion(heaveState, t, out float heave, out float heaveVel, out float heaveAcc);

                writer.WriteLine($"{t:F4},{roll:F6},{pitch:F6},{heave:F6},{rollVel:F6},{pitchVel:F6},{heaveVel:F6},{rollAcc:F6},{pitchAcc:F6},{heaveAcc:F6}");
            }
        }

        UnityEngine.Debug.Log($"[WaveSpectrumTests.TestFFTSpectrumValidation]");
        UnityEngine.Debug.Log($"  已生成 600s @ 100Hz 时序数据");
        UnityEngine.Debug.Log($"  输出文件: {csvPath}");
        UnityEngine.Debug.Log($"  采样点数: {numSamples}");
        UnityEngine.Debug.Log($"  请运行 manuscript/revision/verification/analyze_wave_spectrum.py 进行 FFT 分析");

        Assert.IsTrue(System.IO.File.Exists(csvPath), "时序 CSV 文件应已生成");
    }

    /// <summary>
    /// 辅助函数：计算 RMS
    /// </summary>
    private float ComputeRMS(List<float> values)
    {
        float sumSquares = 0f;
        foreach (float v in values)
            sumSquares += v * v;
        return (float)Math.Sqrt(sumSquares / values.Count);
    }
}
